'use strict';

// Per-pass GPU timing of the Field A physics step. Adds timestamp queries to the step's
// compute passes, so the production step runs unmodified. Reports, per step:
// time of each pass kind, the GPU span from first to last pass (which includes the gaps
// between passes), and the CPU time spent encoding the step.

const QUERY_CAPACITY  = 4096;
const WARMUP_STEPS    = 20;
const PHYSICS_DT      = 1 / 60;
const DISK_MODE_SPIN  = 0;

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}

function pipelineNames(physics) {
  const names = new Map();
  for (const key of Object.keys(physics)) {
    if (key.endsWith('Pipeline')) names.set(physics[key], key.replace(/^_/, '').replace(/Pipeline$/, ''));
  }
  return names;
}

// Patches the encoder prototypes while recording, so it times any pass structure the step
// uses: one pass per dispatch, or many dispatches in one pass. A pass is named after the
// pipelines dispatched in it.
class PassRecorder {
  constructor(device, physics) {
    this._device   = device;
    this._names    = pipelineNames(physics);
    this._querySet = device.createQuerySet({ type: 'timestamp', count: QUERY_CAPACITY });
    this._resolve  = device.createBuffer({ size: QUERY_CAPACITY * 8, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    this._readback = device.createBuffer({ size: QUERY_CAPACITY * 8, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    this._passes   = [];
    this.stepIndex = 0;

    const recorder = this;
    const originalBegin       = GPUCommandEncoder.prototype.beginComputePass;
    const originalSetPipeline = GPUComputePassEncoder.prototype.setPipeline;
    this._restore = () => {
      GPUCommandEncoder.prototype.beginComputePass   = originalBegin;
      GPUComputePassEncoder.prototype.setPipeline    = originalSetPipeline;
    };
    GPUCommandEncoder.prototype.beginComputePass = function (descriptor = {}) {
      if (recorder._passes.length * 2 + 2 > QUERY_CAPACITY) return originalBegin.call(this, descriptor);
      const queryIndex = recorder._passes.length * 2;
      const record = { names: [], step: recorder.stepIndex };
      recorder._passes.push(record);
      const pass = originalBegin.call(this, {
        ...descriptor,
        timestampWrites: { querySet: recorder._querySet, beginningOfPassWriteIndex: queryIndex, endOfPassWriteIndex: queryIndex + 1 },
      });
      pass.__profilerRecord = record;
      return pass;
    };
    GPUComputePassEncoder.prototype.setPipeline = function (pipeline) {
      const name = recorder._names.get(pipeline) ?? 'unknown';
      if (this.__profilerRecord && !this.__profilerRecord.names.includes(name)) this.__profilerRecord.names.push(name);
      return originalSetPipeline.call(this, pipeline);
    };
  }

  stop() {
    this._restore();
  }

  async collect() {
    const queryCount = this._passes.length * 2;
    const encoder = this._device.createCommandEncoder();
    encoder.resolveQuerySet(this._querySet, 0, queryCount, this._resolve, 0);
    encoder.copyBufferToBuffer(this._resolve, 0, this._readback, 0, queryCount * 8);
    this._device.queue.submit([encoder.finish()]);
    await this._readback.mapAsync(GPUMapMode.READ);
    const stamps = new BigInt64Array(this._readback.getMappedRange(0, queryCount * 8).slice(0));
    this._readback.unmap();
    const passes = this._passes.map((pass, index) => ({
      name:    pass.names.join('+') || 'empty',
      step:    pass.step,
      beginNs: stamps[2 * index],
      endNs:   stamps[2 * index + 1],
    }));
    this._passes = [];
    return passes;
  }
}

function summarize(passes, stepCount, cpuEncodeMs) {
  const byName = new Map();
  for (const pass of passes) {
    const entry = byName.get(pass.name) ?? { totalMs: 0, count: 0 };
    entry.totalMs += Number(pass.endNs - pass.beginNs) / 1e6;
    entry.count   += 1;
    byName.set(pass.name, entry);
  }
  const spans = [];
  for (let step = 0; step < stepCount; step++) {
    const stepPasses = passes.filter(pass => pass.step === step);
    if (stepPasses.length === 0) continue;
    const begin = stepPasses.reduce((min, pass) => pass.beginNs < min ? pass.beginNs : min, stepPasses[0].beginNs);
    const end   = stepPasses.reduce((max, pass) => pass.endNs > max ? pass.endNs : max, stepPasses[0].endNs);
    spans.push(Number(end - begin) / 1e6);
  }
  const rows = [...byName.entries()]
    .map(([name, entry]) => ({ name, msPerStep: entry.totalMs / stepCount, passesPerStep: entry.count / stepCount }))
    .sort((a, b) => b.msPerStep - a.msPerStep);
  const passSumMs = rows.reduce((sum, row) => sum + row.msPerStep, 0);
  return { rows, passSumMs, gpuSpanMs: median(spans), cpuEncodeMs: median(cpuEncodeMs) };
}

export async function profileFieldAStep({ device, physics, steps, log }) {
  physics.injectDisk([0.5, 0.5], 0.35, 100, DISK_MODE_SPIN);
  for (let step = 0; step < WARMUP_STEPS; step++) physics.step(PHYSICS_DT);
  await device.queue.onSubmittedWorkDone();

  const recorder = new PassRecorder(device, physics);
  const cpuEncodeMs = [];
  for (let step = 0; step < steps; step++) {
    recorder.stepIndex = step;
    const start = performance.now();
    physics.step(PHYSICS_DT);
    cpuEncodeMs.push(performance.now() - start);
    // One step per submit-and-wait keeps the per-step GPU span free of queued work.
    await device.queue.onSubmittedWorkDone();
  }
  recorder.stop();

  const wallStart = performance.now();
  for (let step = 0; step < steps; step++) physics.step(PHYSICS_DT);
  await device.queue.onSubmittedWorkDone();
  const pipelinedMs = (performance.now() - wallStart) / steps;

  const summary = summarize(await recorder.collect(), steps, cpuEncodeMs);
  for (const row of summary.rows) {
    log(`  ${row.name.padEnd(16)} ${row.msPerStep.toFixed(3).padStart(8)} ms/step  (${row.passesPerStep} passes, ${(row.msPerStep / row.passesPerStep * 1000).toFixed(1)} µs each)`);
  }
  log(`  sum of passes    ${summary.passSumMs.toFixed(3).padStart(8)} ms`);
  log(`  GPU span         ${summary.gpuSpanMs.toFixed(3).padStart(8)} ms  (gaps between passes: ${(summary.gpuSpanMs - summary.passSumMs).toFixed(3)} ms)`);
  log(`  CPU encode       ${summary.cpuEncodeMs.toFixed(3).padStart(8)} ms  (step() call)`);
  log(`  pipelined wall   ${pipelinedMs.toFixed(3).padStart(8)} ms/step  (steps submitted back to back)`);
  return { ...summary, pipelinedMs };
}

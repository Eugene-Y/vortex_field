'use strict';

import { VELOCITY_SCENARIOS } from './VelocityScenarios.js';
import { buildVariantShader, buildReduceShader, accumulationWordsPerCell, tileShape } from './RotationKernelGenerator.js';

// Times Field B accumulation variants on synthetic velocity fields and checks each
// variant's output against the production shader (src/shaders/rotation_compute.wgsl).
// One frame = clearBuffer(accumulation) + compute pass + reduce pass, exactly as the app.

const WORKGROUP_SIZE          = 64;
const ROTATION_REFERENCE_GRID = 100;
const PARALLEL_THRESHOLD      = 0.001;
const TARGET_BATCH_MS         = 400;
const MAX_FRAMES_PER_BATCH    = 60;
const BATCH_COUNT             = 5;
const PARAMS_BYTE_SIZE        = 64;
// Pathological contention can take seconds per frame; one probe frame is enough there.
const SLOW_FRAME_MS           = 150;

export const PRODUCTION_VARIANT = { name: 'production', production: true, accumulationBuffers: 32 };

function buildAnnulusOffsets(gridSize, pairDistance, distanceDelta) {
  const halfGrid    = gridSize / 2;
  const minRadius   = Math.max(0, (pairDistance - distanceDelta) * halfGrid);
  const maxRadius   = (pairDistance + distanceDelta) * halfGrid;
  const minRadiusSq = minRadius * minRadius;
  const maxRadiusSq = maxRadius * maxRadius;
  const full = [];
  const iMax = Math.min(Math.ceil(maxRadius), halfGrid);
  for (let dRow = -iMax; dRow <= iMax; dRow++) {
    for (let dCol = -iMax; dCol <= iMax; dCol++) {
      const distSq = dRow * dRow + dCol * dCol;
      if (distSq >= minRadiusSq && distSq <= maxRadiusSq) full.push([dCol, dRow]);
    }
  }
  const halfPlane = full.filter(([dCol, dRow]) => dRow > 0 || (dRow === 0 && dCol > 0));
  return { full, halfPlane, rowSpans: groupIntoRowSpans(halfPlane), minRadius, maxRadius };
}

// [dRow, dColStart, dColEnd, unused] runs of consecutive dCol; covers exactly the input offsets.
function groupIntoRowSpans(offsets) {
  const spans = [];
  for (const [dCol, dRow] of offsets) {
    const last = spans[spans.length - 1];
    if (last && last[0] === dRow && last[2] === dCol - 1) last[2] = dCol;
    else spans.push([dRow, dCol, dCol, 0]);
  }
  return spans;
}

function writeComputeParams(device, buffer, config, offsetCount, minRadius, maxRadius) {
  const raw = new ArrayBuffer(PARAMS_BYTE_SIZE);
  const u32 = new Uint32Array(raw);
  const f32 = new Float32Array(raw);
  const stride = config.sampleStride ?? 1;
  const mask = config.mask;
  u32[0]  = config.gridSize;
  f32[1]  = stride * stride * config.gridSize / ROTATION_REFERENCE_GRID;
  f32[2]  = PARALLEL_THRESHOLD;
  u32[3]  = offsetCount;
  f32[4]  = minRadius;
  f32[5]  = maxRadius;
  u32[6]  = stride;
  u32[7]  = mask ? mask.originX : 0;
  u32[8]  = mask ? mask.originY : 0;
  u32[9]  = mask ? mask.boxSize : 0;
  f32[10] = mask ? mask.centerX : 0;
  f32[11] = mask ? mask.centerY : 0;
  f32[12] = mask ? mask.radius * mask.radius : 0;
  u32[13] = mask ? 1 : 0;
  u32[14] = config.boundaryMode ?? 0;
  u32[15] = config.showMask ?? 3;
  device.queue.writeBuffer(buffer, 0, raw);
}

function resolveMaskBox(gridSize, mask) {
  if (!mask) return null;
  const boxSize = Math.min(gridSize, Math.ceil(2 * mask.radius + 2));
  const halfBox = Math.floor(boxSize / 2);
  const originX = Math.max(0, Math.min(gridSize - boxSize, Math.round(mask.centerX) - halfBox));
  const originY = Math.max(0, Math.min(gridSize - boxSize, Math.round(mask.centerY) - halfBox));
  return { ...mask, boxSize, originX, originY };
}

function createVelocityTexture(device, gridSize, velocity) {
  const texture = device.createTexture({
    size:   [gridSize, gridSize],
    format: 'rg32float',
    usage:  GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC,
  });
  device.queue.writeTexture({ texture }, velocity, { bytesPerRow: gridSize * 8 }, [gridSize, gridSize]);
  return texture;
}

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}

function compareOutputs(candidate, reference) {
  let absoluteErrorSum = 0;
  let referenceSum     = 0;
  let maxError         = 0;
  let maxReference     = 0;
  for (let index = 0; index < reference.length; index++) {
    const error = Math.abs(candidate[index] - reference[index]);
    absoluteErrorSum += error;
    referenceSum     += Math.abs(reference[index]);
    maxError          = Math.max(maxError, error);
    maxReference      = Math.max(maxReference, Math.abs(reference[index]));
  }
  return {
    relativeL1:  referenceSum > 0 ? absoluteErrorSum / referenceSum : absoluteErrorSum,
    relativeMax: maxReference > 0 ? maxError / maxReference : maxError,
  };
}

class VariantRuntime {
  constructor(device, variant, gridSize, productionSources) {
    this._device   = device;
    this.variant   = variant;
    const totalCells   = gridSize * gridSize;
    const accumulation = variant.production ? 'casF32'
      : variant.aggregation === 'workgroup' ? 'fixed64' : variant.accumulation;
    this.accumulationBuffers = variant.accumulationBuffers;
    this.offsetSplit = variant.offsetSplit ?? 1;
    this.offsetTable = variant.production ? 'full' : variant.offsetTable;
    this.cellOrder   = variant.cellOrder ?? 'rowMajor';
    this.workgroupSize = variant.workgroupSize ?? WORKGROUP_SIZE;
    this.velocitySource = variant.velocitySource ?? 'texture';

    const constants = { ACCUMULATION_BUFFERS: variant.accumulationBuffers };
    if (accumulation === 'fixed32' || accumulation === 'fixed64') constants.FIXED_POINT_SCALE = variant.fixedPointScale;
    const computeSource = variant.production ? productionSources.compute : buildVariantShader(variant);
    const reduceSource  = variant.production ? productionSources.reduce  : buildReduceShader(accumulation);

    this.computePipeline = device.createComputePipeline({
      layout: 'auto',
      compute: { module: device.createShaderModule({ code: computeSource }), entryPoint: 'computeRotation', constants },
    });
    const reduceConstants = constants.FIXED_POINT_SCALE ? { FIXED_POINT_SCALE: constants.FIXED_POINT_SCALE } : {};
    this.reducePipeline = device.createComputePipeline({
      layout: 'auto',
      compute: { module: device.createShaderModule({ code: reduceSource }), entryPoint: 'reduceBuffers', constants: reduceConstants },
    });

    this.accumulationBuffer = device.createBuffer({
      size:  variant.accumulationBuffers * totalCells * 4 * accumulationWordsPerCell(accumulation),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
  }

  dispose() {
    this.accumulationBuffer.destroy();
  }
}

class BenchmarkCase {
  constructor(device, config, velocity, timestampSupported) {
    this._device   = device;
    this.config    = config;
    this.gridSize  = config.gridSize;
    this.totalCells = config.gridSize * config.gridSize;
    this._timestampSupported = timestampSupported;

    this.velocityTexture = createVelocityTexture(device, config.gridSize, velocity);
    this.annulus = buildAnnulusOffsets(config.gridSize, config.pairDistance, config.distanceDelta);
    this.mask    = resolveMaskBox(config.gridSize, config.mask);

    this.offsetBuffers = {
      full:      this._uploadOffsets(this.annulus.full),
      halfPlane: this._uploadOffsets(this.annulus.halfPlane),
      rowSpans:  this._uploadOffsets(this.annulus.rowSpans),
    };
    this.paramsBuffers = {};
    for (const table of ['full', 'halfPlane', 'rowSpans']) {
      this.paramsBuffers[table] = device.createBuffer({ size: PARAMS_BYTE_SIZE, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      writeComputeParams(device, this.paramsBuffers[table], { ...config, mask: this.mask },
        this.annulus[table].length, this.annulus.minRadius, this.annulus.maxRadius);
    }

    this.outputBuffer = device.createBuffer({
      size: this.totalCells * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    this.readbackBuffer = device.createBuffer({
      size: this.totalCells * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    this.velocityStorageBuffer = device.createBuffer({
      size: this.totalCells * 8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
  }

  _uploadOffsets(entries) {
    const flat = new Int32Array(Math.max(4, entries.flat().length));
    flat.set(entries.flat());
    const buffer = this._device.createBuffer({ size: flat.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this._device.queue.writeBuffer(buffer, 0, flat);
    return buffer;
  }

  bindRuntime(runtime) {
    const device = this._device;
    const reduceParams = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(reduceParams, 0, new Uint32Array([this.gridSize, runtime.accumulationBuffers, 0, 0]));
    return {
      runtime,
      computeBindGroup: device.createBindGroup({
        layout: runtime.computePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: runtime.velocitySource === 'storage'
            ? { buffer: this.velocityStorageBuffer } : this.velocityTexture.createView() },
          { binding: 1, resource: { buffer: runtime.accumulationBuffer } },
          { binding: 2, resource: { buffer: this.paramsBuffers[runtime.offsetTable] } },
          { binding: 3, resource: { buffer: this.offsetBuffers[runtime.offsetTable] } },
        ],
      }),
      reduceBindGroup: device.createBindGroup({
        layout: runtime.reducePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: runtime.accumulationBuffer } },
          { binding: 1, resource: { buffer: this.outputBuffer } },
          { binding: 2, resource: { buffer: reduceParams } },
        ],
      }),
    };
  }

  _computeWorkgroupCount(runtime) {
    const workgroupSize = runtime.workgroupSize;
    if (this.mask) return Math.ceil(this.mask.boxSize * this.mask.boxSize / workgroupSize);
    if (runtime.cellOrder === 'tiled8') {
      const tile = tileShape(workgroupSize);
      return Math.ceil(this.gridSize / tile.width) * Math.ceil(this.gridSize / tile.height);
    }
    return Math.ceil(this.totalCells / workgroupSize);
  }

  encodeFrame(encoder, bound, timestampWrites) {
    if (bound.runtime.velocitySource === 'storage') {
      encoder.copyTextureToBuffer({ texture: this.velocityTexture },
        { buffer: this.velocityStorageBuffer, bytesPerRow: this.gridSize * 8 }, [this.gridSize, this.gridSize]);
    }
    encoder.clearBuffer(bound.runtime.accumulationBuffer);
    const computePass = encoder.beginComputePass(timestampWrites ? { timestampWrites: timestampWrites.compute } : {});
    computePass.setPipeline(bound.runtime.computePipeline);
    computePass.setBindGroup(0, bound.computeBindGroup);
    computePass.dispatchWorkgroups(this._computeWorkgroupCount(bound.runtime), bound.runtime.offsetSplit);
    computePass.end();
    const reducePass = encoder.beginComputePass(timestampWrites ? { timestampWrites: timestampWrites.reduce } : {});
    reducePass.setPipeline(bound.runtime.reducePipeline);
    reducePass.setBindGroup(0, bound.reduceBindGroup);
    reducePass.dispatchWorkgroups(Math.ceil(this.totalCells / WORKGROUP_SIZE));
    reducePass.end();
  }

  async readOutput(bound) {
    const encoder = this._device.createCommandEncoder();
    this.encodeFrame(encoder, bound, null);
    encoder.copyBufferToBuffer(this.outputBuffer, 0, this.readbackBuffer, 0, this.totalCells * 4);
    this._device.queue.submit([encoder.finish()]);
    await this.readbackBuffer.mapAsync(GPUMapMode.READ);
    const output = new Float32Array(this.readbackBuffer.getMappedRange().slice(0));
    this.readbackBuffer.unmap();
    return output;
  }

  // Metal can only allocate a few counter sample buffers, so one query set is reused.
  _ensureTimestampResources() {
    if (!this._timestampSupported || this._querySet) return;
    const device = this._device;
    const queryCount = MAX_FRAMES_PER_BATCH * 4;
    this._querySet = device.createQuerySet({ type: 'timestamp', count: queryCount });
    this._resolveBuffer = device.createBuffer({ size: queryCount * 8, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    this._timestampReadback = device.createBuffer({ size: queryCount * 8, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  }

  async _timeBatch(bound, frameCount) {
    const device = this._device;
    this._ensureTimestampResources();
    const querySet = this._querySet ?? null;
    const resolveBuffer = this._resolveBuffer;
    const timestampReadback = this._timestampReadback;
    const encoder = device.createCommandEncoder();
    for (let frame = 0; frame < frameCount; frame++) {
      const timestampWrites = querySet ? {
        compute: { querySet, beginningOfPassWriteIndex: 4 * frame,     endOfPassWriteIndex: 4 * frame + 1 },
        reduce:  { querySet, beginningOfPassWriteIndex: 4 * frame + 2, endOfPassWriteIndex: 4 * frame + 3 },
      } : null;
      this.encodeFrame(encoder, bound, timestampWrites);
    }
    if (querySet) {
      encoder.resolveQuerySet(querySet, 0, frameCount * 4, resolveBuffer, 0);
      encoder.copyBufferToBuffer(resolveBuffer, 0, timestampReadback, 0, frameCount * 32);
    }
    const start = performance.now();
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    const wallMs = (performance.now() - start) / frameCount;

    let computeMs = NaN;
    let reduceMs  = NaN;
    if (querySet) {
      await timestampReadback.mapAsync(GPUMapMode.READ);
      const stamps = new BigInt64Array(timestampReadback.getMappedRange(0, frameCount * 32).slice(0));
      timestampReadback.unmap();
      const computeTimes = [];
      const reduceTimes  = [];
      for (let frame = 0; frame < frameCount; frame++) {
        computeTimes.push(Number(stamps[4 * frame + 1] - stamps[4 * frame]) / 1e6);
        reduceTimes.push(Number(stamps[4 * frame + 3] - stamps[4 * frame + 2]) / 1e6);
      }
      computeMs = median(computeTimes);
      reduceMs  = median(reduceTimes);
    }
    return { wallMs, computeMs, reduceMs };
  }

  async time(bound) {
    const probe = await this._timeBatch(bound, 1);
    if (probe.wallMs > SLOW_FRAME_MS) return { ...probe, minComputeMs: probe.computeMs };
    const frameCount = Math.max(2, Math.min(MAX_FRAMES_PER_BATCH, Math.ceil(TARGET_BATCH_MS / Math.max(probe.wallMs, 0.05))));
    const batches = [];
    for (let batch = 0; batch < BATCH_COUNT; batch++) batches.push(await this._timeBatch(bound, frameCount));
    return {
      wallMs:       median(batches.map(result => result.wallMs)),
      computeMs:    median(batches.map(result => result.computeMs)),
      reduceMs:     median(batches.map(result => result.reduceMs)),
      minComputeMs: Math.min(...batches.map(result => result.computeMs)),
    };
  }

  dispose() {
    this.velocityTexture.destroy();
    Object.values(this.offsetBuffers).forEach(buffer => buffer.destroy());
    Object.values(this.paramsBuffers).forEach(buffer => buffer.destroy());
    this.outputBuffer.destroy();
    this.readbackBuffer.destroy();
    this.velocityStorageBuffer.destroy();
    this._querySet?.destroy();
    this._resolveBuffer?.destroy();
    this._timestampReadback?.destroy();
  }
}

export async function runRotationBenchmark({ device, productionSources, configs, scenarios, variants, log, timestampSupported, loadFieldFile }) {
  const results = [];
  for (const config of configs) {
    for (const scenarioName of scenarios) {
      const velocity = VELOCITY_SCENARIOS[scenarioName]
        ? VELOCITY_SCENARIOS[scenarioName](config.gridSize)
        : new Float32Array(await loadFieldFile(`${scenarioName}${config.gridSize}.f32`));
      const benchCase = new BenchmarkCase(device, config, velocity, timestampSupported);
      const pairCount = benchCase.annulus.halfPlane.length;
      const spanCount = benchCase.annulus.rowSpans.length;
      log(`\n## ${config.name} · ${scenarioName} — grid ${config.gridSize}, ${benchCase.annulus.full.length} offsets (${pairCount} unique per cell, ${spanCount} row spans)`);

      const productionRuntime = new VariantRuntime(device, PRODUCTION_VARIANT, config.gridSize, productionSources);
      const productionBound   = benchCase.bindRuntime(productionRuntime);
      const reference = await benchCase.readOutput(productionBound);
      const rerun     = await benchCase.readOutput(productionBound);
      const noiseFloor = compareOutputs(rerun, reference);
      let referenceAbsSum = 0;
      let referenceMax = 0;
      for (const value of reference) { referenceAbsSum += Math.abs(value); referenceMax = Math.max(referenceMax, Math.abs(value)); }
      log(`reference: mean|x| ${(referenceAbsSum / reference.length).toExponential(2)}, max|x| ${referenceMax.toExponential(2)}, run-to-run relL1 ${noiseFloor.relativeL1.toExponential(1)}`);
      productionRuntime.dispose();

      let productionWall = null;
      for (const variant of variants) {
        device.pushErrorScope('validation');
        const runtime = new VariantRuntime(device, variant, config.gridSize, productionSources);
        const buildError = await device.popErrorScope();
        if (buildError) {
          log(`${variant.name.padEnd(34)} build failed: ${buildError.message.split('\n').slice(0, 4).join(' ')}`);
          runtime.dispose();
          continue;
        }
        const bound  = benchCase.bindRuntime(runtime);
        const output = await benchCase.readOutput(bound);
        const error  = compareOutputs(output, reference);
        const timing = await benchCase.time(bound);
        if (variant.production && productionWall === null) productionWall = timing.computeMs || timing.wallMs;
        const speedup = productionWall ? productionWall / (timing.computeMs || timing.wallMs) : NaN;
        const gigaPairsPerSecond = pairCount * benchCase.totalCells / (timing.computeMs || timing.wallMs) / 1e6;
        log(`${variant.name.padEnd(34)} frame ${timing.wallMs.toFixed(3).padStart(8)} ms` +
          `  compute ${timing.computeMs.toFixed(3).padStart(8)} (min ${timing.minComputeMs.toFixed(3).padStart(8)})  reduce ${timing.reduceMs.toFixed(3).padStart(6)}` +
          `  ×${speedup.toFixed(2).padStart(5)}  ${gigaPairsPerSecond.toFixed(2).padStart(6)} Gpair/s` +
          `  relL1 ${error.relativeL1.toExponential(1)}  relMax ${error.relativeMax.toExponential(1)}`);
        results.push({ config: config.name, scenario: scenarioName, variant: variant.name, ...timing, speedup, ...error });
        runtime.dispose();
        await device.queue.onSubmittedWorkDone();
      }
      benchCase.dispose();
    }
  }
  return results;
}

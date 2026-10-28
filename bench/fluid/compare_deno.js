'use strict';

// deno run --unstable-webgpu --allow-read bench/fluid/compare_deno.js [query]
//
// Runs the reference Field A step (bench/fluid/reference, copied from main) and the current
// src/ step side by side from the same initial flow: checks the velocity fields stay equal
// and times both in alternating rounds, so background GPU load hits them alike.

const query = Deno.args[0] ?? '';
globalThis.window = globalThis;
globalThis.window.location = { search: `?${query}`, origin: '', pathname: '' };

const repoRoot = new URL('../../', import.meta.url);
const readRepoText = path => Deno.readTextFile(new URL(path, repoRoot));

const { WebGPUNavierStokesStep } = await import('../../src/simulation/WebGPUNavierStokesStep.js');
const { WebGPUNavierStokesStep: ReferenceStep } = await import('./reference/WebGPUNavierStokesStep.js');
const { GRID_SIZE, PHYSICS_DEFAULTS } = await import('../../src/config/SimulationConfig.js');
const { loadShaderSources, createHeadlessCanvas, readVelocity, SHADER_FILES } = await import('./HeadlessPhysics.js');

const PHYSICS_DT        = 1 / 60;
const EQUIVALENCE_STEPS = Number(Deno.args[1] ?? 120);
const TIMING_ROUNDS     = Number(Deno.args[2] ?? 6);
const STEPS_PER_ROUND   = 30;
const DISK_MODE_SPIN    = 0;
const NOISE_SEED        = 7;

const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
const device  = await adapter.requestDevice();
device.addEventListener('uncapturederror', event => console.error('GPU error:', event.error.message));

const referenceShaderNames = (await Array.fromAsync(Deno.readDir(new URL('reference/shaders/', import.meta.url))))
  .map(entry => entry.name.replace(/\.wgsl$/, ''));
const referenceDirectories = Object.fromEntries(referenceShaderNames.map(name => [name, 'bench/fluid/reference/shaders']));

const steps = {
  reference: new ReferenceStep(device, createHeadlessCanvas(device), await loadShaderSources(readRepoText, referenceDirectories)),
  current:   new WebGPUNavierStokesStep(device, createHeadlessCanvas(device), await loadShaderSources(readRepoText)),
};

for (const physics of Object.values(steps)) {
  physics.injectDisk([0.45, 0.5], 0.3, 100, DISK_MODE_SPIN);
  physics.injectNoise(30, NOISE_SEED);
}

console.log(`grid ${GRID_SIZE}, pressure iterations ${PHYSICS_DEFAULTS.pressureIterations}, boundary ${PHYSICS_DEFAULTS.boundaryMode}`);

for (let step = 0; step < EQUIVALENCE_STEPS; step++) {
  steps.reference.step(PHYSICS_DT);
  steps.current.step(PHYSICS_DT);
}
const referenceVelocity = await readVelocity(device, steps.reference.velocityTexture);
const currentVelocity   = await readVelocity(device, steps.current.velocityTexture);
let maxDifference = 0;
let maxMagnitude  = 0;
let differingValues = 0;
for (let index = 0; index < referenceVelocity.length; index++) {
  const difference = Math.abs(currentVelocity[index] - referenceVelocity[index]);
  if (difference > 0) differingValues++;
  maxDifference = Math.max(maxDifference, difference);
  maxMagnitude  = Math.max(maxMagnitude, Math.abs(referenceVelocity[index]));
}
console.log(`after ${EQUIVALENCE_STEPS} steps: ${differingValues} of ${referenceVelocity.length} values differ, ` +
  `max |Δv| ${maxDifference.toExponential(2)} (max |v| ${maxMagnitude.toFixed(1)})`);

async function timeRound(physics) {
  const start = performance.now();
  for (let step = 0; step < STEPS_PER_ROUND; step++) physics.step(PHYSICS_DT);
  await device.queue.onSubmittedWorkDone();
  return (performance.now() - start) / STEPS_PER_ROUND;
}

const timings = { reference: [], current: [] };
for (let round = 0; round < TIMING_ROUNDS; round++) {
  for (const [name, physics] of Object.entries(steps)) timings[name].push(await timeRound(physics));
}
const median = values => [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)];
const referenceMs = median(timings.reference);
const currentMs   = median(timings.current);
console.log(`step, steps back to back: reference ${referenceMs.toFixed(3)} ms, current ${currentMs.toFixed(3)} ms, ×${(referenceMs / currentMs).toFixed(2)}`);
console.log(`  rounds reference: ${timings.reference.map(ms => ms.toFixed(2)).join(' ')}`);
console.log(`  rounds current:   ${timings.current.map(ms => ms.toFixed(2)).join(' ')}`);

'use strict';

// deno run --unstable-webgpu --allow-read bench/fluid/run_deno.js [query]
// query: the app's URL parameters, e.g. "gridSize=512&pressureIters=20" (default: app defaults).

const query = Deno.args[0] ?? '';
globalThis.window = globalThis;
globalThis.window.location = { search: `?${query}`, origin: '', pathname: '' };

const repoRoot = new URL('../../', import.meta.url);
const readRepoText = path => Deno.readTextFile(new URL(path, repoRoot));

const { WebGPUNavierStokesStep } = await import('../../src/simulation/WebGPUNavierStokesStep.js');
const { GRID_SIZE, PHYSICS_DEFAULTS } = await import('../../src/config/SimulationConfig.js');
const { profileFieldAStep } = await import('./FieldAProfiler.js');

const PROFILED_STEPS = 20;

const SHADER_FILES = {
  advect: 'velocity_advect', divergence: 'velocity_divergence', pressure: 'velocity_pressure',
  subtractGradient: 'velocity_subtract_gradient', vorticityCurl: 'velocity_vorticity_curl',
  vorticityConfinement: 'velocity_vorticity_confinement', injectImpulse: 'velocity_inject_impulse',
  injectDisk: 'velocity_inject_disk', noise: 'velocity_noise', render: 'velocity_render',
  densityAdvect: 'density_advect', gravityPoisson: 'gravity_poisson', gravityForce: 'gravity_force',
  densityInjectImpulse: 'density_inject_impulse', densityInjectDisk: 'density_inject_disk',
  densityInjectNoise: 'density_inject_noise',
};

function createHeadlessCanvas(device) {
  return {
    getContext: () => ({
      configure: () => {},
      getCurrentTexture: () => device.createTexture({
        size: [1, 1], format: navigator.gpu.getPreferredCanvasFormat(), usage: GPUTextureUsage.RENDER_ATTACHMENT,
      }),
    }),
  };
}

const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
const device  = await adapter.requestDevice({ requiredFeatures: ['timestamp-query'] });
device.addEventListener('uncapturederror', event => console.error('GPU error:', event.error.message));

const shaderSources = {};
for (const [key, file] of Object.entries(SHADER_FILES)) shaderSources[key] = await readRepoText(`src/shaders/${file}.wgsl`);

const physics = new WebGPUNavierStokesStep(device, createHeadlessCanvas(device), shaderSources);
console.log(`grid ${GRID_SIZE}, pressure iterations ${PHYSICS_DEFAULTS.pressureIterations}, vorticity ${PHYSICS_DEFAULTS.vorticityStrength}, boundary ${PHYSICS_DEFAULTS.boundaryMode}`);
await profileFieldAStep({ device, physics, steps: PROFILED_STEPS, log: line => console.log(line) });

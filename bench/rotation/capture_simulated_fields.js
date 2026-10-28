'use strict';

// deno run --unstable-webgpu --allow-read --allow-write bench/rotation/capture_simulated_fields.js [gridSize]
//
// Runs the production Navier–Stokes step headless with the app defaults and saves raw
// rg32float velocity snapshots to bench/rotation/fields/, so the benchmark can also be
// driven by flows the app actually produces, not only by synthetic ones.

const gridSize = Number(Deno.args[0] ?? 256);
globalThis.window = globalThis;
globalThis.window.location = { search: `?gridSize=${gridSize}`, origin: '', pathname: '' };

const repoRoot = new URL('../../', import.meta.url);
const readRepoText = path => Deno.readTextFile(new URL(path, repoRoot));

const { WebGPUNavierStokesStep } = await import('../../src/simulation/WebGPUNavierStokesStep.js');
const { MOUSE_DEFAULTS } = await import('../../src/config/SimulationConfig.js');

const PHYSICS_DT          = 1 / 60;
const SETTLE_STEPS        = 600;
const DISK_MODE_SPIN      = 0;
const DISK_MODE_EXPLODE   = 1;
const DISK_MODE_IMPLODE   = 2;

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

// The physics textures lack COPY_SRC, so a compute pass copies them into a buffer.
const TEXTURE_TO_BUFFER_SHADER = /* wgsl */`
@group(0) @binding(0) var velocityTexture: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> velocityBuffer: array<vec2f>;
@compute @workgroup_size(8, 8)
fn copyVelocity(@builtin(global_invocation_id) id: vec3<u32>) {
  let size = textureDimensions(velocityTexture);
  if (id.x >= size.x || id.y >= size.y) { return; }
  velocityBuffer[id.y * size.x + id.x] = textureLoad(velocityTexture, vec2i(id.xy), 0).xy;
}
`;

async function readVelocity(device, texture) {
  const byteSize = gridSize * gridSize * 8;
  const storage  = device.createBuffer({ size: byteSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const readback = device.createBuffer({ size: byteSize, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const pipeline = device.createComputePipeline({
    layout: 'auto',
    compute: { module: device.createShaderModule({ code: TEXTURE_TO_BUFFER_SHADER }), entryPoint: 'copyVelocity' },
  });
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: texture.createView() }, { binding: 1, resource: { buffer: storage } }],
  });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(gridSize / 8), Math.ceil(gridSize / 8));
  pass.end();
  encoder.copyBufferToBuffer(storage, 0, readback, 0, byteSize);
  device.queue.submit([encoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  const velocity = new Float32Array(readback.getMappedRange().slice(0));
  readback.unmap();
  storage.destroy();
  readback.destroy();
  return velocity;
}

async function simulate(device, shaderSources, inject) {
  const physics = new WebGPUNavierStokesStep(device, createHeadlessCanvas(device), shaderSources);
  inject(physics);
  for (let step = 0; step < SETTLE_STEPS; step++) physics.step(PHYSICS_DT);
  const velocity = await readVelocity(device, physics.velocityTexture);
  physics.dispose?.();
  return velocity;
}

const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
const device  = await adapter.requestDevice();
device.addEventListener('uncapturederror', event => console.error('GPU error:', event.error.message));
const shaderSources = {};
for (const [key, file] of Object.entries(SHADER_FILES)) shaderSources[key] = await readRepoText(`src/shaders/${file}.wgsl`);

const patternRadius = MOUSE_DEFAULTS.patternScale * 0.5;
const strength      = MOUSE_DEFAULTS.impulseStrength;

const captures = {
  simulatedSpin: physics => physics.injectDisk([0.5, 0.5], patternRadius, strength, DISK_MODE_SPIN),
  simulatedMixed: physics => {
    physics.injectDisk([0.5, 0.5], patternRadius, strength, DISK_MODE_SPIN);
    physics.injectDisk([0.25, 0.3], 0.15, strength, DISK_MODE_EXPLODE);
    physics.injectDisk([0.75, 0.7], 0.12, -strength, DISK_MODE_SPIN);
    physics.injectDisk([0.7, 0.25], 0.1, strength, DISK_MODE_IMPLODE);
    physics.injectNoise(strength * 0.3, 7);
  },
};

const outputDirectory = new URL('fields/', import.meta.url);
await Deno.mkdir(outputDirectory, { recursive: true });
for (const [name, inject] of Object.entries(captures)) {
  const velocity = await simulate(device, shaderSources, inject);
  let maxSpeed = 0;
  for (let index = 0; index < velocity.length; index += 2) maxSpeed = Math.max(maxSpeed, Math.hypot(velocity[index], velocity[index + 1]));
  await Deno.writeFile(new URL(`${name}${gridSize}.f32`, outputDirectory), new Uint8Array(velocity.buffer));
  console.log(`${name}${gridSize}: max speed ${maxSpeed.toFixed(1)}`);
}

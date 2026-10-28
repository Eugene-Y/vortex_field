'use strict';

// Shared setup for running the Field A physics step outside the app.

export const SHADER_FILES = {
  advect: 'velocity_advect', divergence: 'velocity_divergence', pressure: 'velocity_pressure',
  subtractGradient: 'velocity_subtract_gradient', vorticityCurl: 'velocity_vorticity_curl',
  vorticityConfinement: 'velocity_vorticity_confinement', injectImpulse: 'velocity_inject_impulse',
  injectDisk: 'velocity_inject_disk', noise: 'velocity_noise', render: 'velocity_render',
  densityAdvect: 'density_advect', gravityPoisson: 'gravity_poisson', gravityForce: 'gravity_force',
  densityInjectImpulse: 'density_inject_impulse', densityInjectDisk: 'density_inject_disk',
  densityInjectNoise: 'density_inject_noise',
};

// readText(path) reads a repo-relative path; shaderDirectories maps a shader file name to
// the directory it is read from when it should not come from src/shaders.
export async function loadShaderSources(readText, shaderDirectories = {}) {
  const sources = {};
  for (const [key, file] of Object.entries(SHADER_FILES)) {
    const directory = shaderDirectories[file] ?? 'src/shaders';
    sources[key] = await readText(`${directory}/${file}.wgsl`);
  }
  return sources;
}

export function createHeadlessCanvas(device) {
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

export async function readVelocity(device, texture) {
  const gridSize = texture.width;
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

'use strict';

import {
  GRID_SIZE, PHYSICS_DEFAULTS, RENDER_DEFAULTS,
} from '../config/SimulationConfig.js';
import { UniformWriter } from '../gpu/UniformWriter.js';

const WORKGROUP_SIZE = 8; // threads per dimension; dispatch is ceil(N/8) × ceil(N/8)

/**
 * Physics step implementation: incompressible Navier-Stokes, fully on WebGPU.
 *
 * Interface contract (all physics steps must implement):
 *   step(deltaTime)
 *   injectImpulse(position, direction, radius, strength)
 *   injectDisk(center, radiusUv, strength, mode)
 *   injectNoise(strength, seed)
 *   clearVelocity()
 *   renderToCanvas()
 *   get velocityTexture   — current GPUTexture (rg32float)
 *   dispose()
 *
 * Texture convention: row 0 = physical bottom (y-up).
 * The render shader maps uv.y=0 (screen bottom) to row 0, preserving orientation.
 */
export class WebGPUNavierStokesStep {
  constructor(device, canvas, shaderSources) {
    this._device   = device;
    this._gridSize = GRID_SIZE;
    this._workgroupsXY = Math.ceil(GRID_SIZE / WORKGROUP_SIZE);

    // Configure WebGPU context on the velocity canvas.
    this._context      = canvas.getContext('webgpu');
    this._canvasFormat = navigator.gpu.getPreferredCanvasFormat();
    this._context.configure({ device, format: this._canvasFormat, alphaMode: 'opaque' });

    // Ping-pong textures for velocity (rg32float) and pressure (r32float).
    this._velocityTextures = [this._createRgTexture(), this._createRgTexture()];
    this._pressureTextures = [this._createRTexture(),  this._createRTexture()];
    this._divergenceTexture = this._createRTexture();
    this._curlTexture       = this._createRTexture();
    this._velocityReadIdx   = 0;
    this._pressureReadIdx   = 0;

    // Zero-filled arrays for clearing textures (allocated once, reused).
    this._velocityZeros = new Float32Array(GRID_SIZE * GRID_SIZE * 2);
    this._pressureZeros = new Float32Array(GRID_SIZE * GRID_SIZE);

    // Uniform buffers.
    this._advectParamsBuffer      = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this._baseParamsBuffer        = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this._confinementParamsBuffer = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this._impulseParamsBuffer     = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this._diskParamsBuffer        = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this._noiseParamsBuffer       = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this._renderParamsBuffer      = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

    // Build compute and render pipelines.
    this._advectPipeline      = this._buildComputePipeline(shaderSources.advect,              'advectVelocity');
    this._divergencePipeline  = this._buildComputePipeline(shaderSources.divergence,          'computeDivergence');
    this._pressurePipeline    = this._buildComputePipeline(shaderSources.pressure,            'jacobiPressureStep');
    this._subtractGradPipeline = this._buildComputePipeline(shaderSources.subtractGradient,   'subtractPressureGradient');
    this._curlPipeline        = this._buildComputePipeline(shaderSources.vorticityCurl,       'computeCurl');
    this._confinePipeline     = this._buildComputePipeline(shaderSources.vorticityConfinement,'applyVorticityConfinement');
    this._impulsePipeline     = this._buildComputePipeline(shaderSources.injectImpulse,       'injectGaussianImpulse');
    this._diskPipeline        = this._buildComputePipeline(shaderSources.injectDisk,          'injectDisk');
    this._noisePipeline       = this._buildComputePipeline(shaderSources.noise,               'injectNoise');
    this._renderPipeline      = this._buildRenderPipeline(shaderSources.render, this._canvasFormat);

    // Initialize all textures to zero.
    this._clearAllTextures();
    // Write static render params (updated each frame for tone midpoint changes).
    this._writeRenderParams();
  }

  // ─── Public interface ────────────────────────────────────────────────────────

  get velocityTexture() {
    return this._velocityTextures[this._velocityReadIdx];
  }

  /** Advances the simulation by deltaTime seconds. */
  step(deltaTime) {
    this._writeAdvectParams(deltaTime);
    this._writeBaseParams();

    const encoder = this._device.createCommandEncoder();

    this._dispatchCompute(encoder, this._advectPipeline,  this._buildAdvectBindGroup());
    this._swapVelocity();

    if (PHYSICS_DEFAULTS.vorticityStrength > 0) {
      this._writeConfinementParams(deltaTime);
      this._dispatchCompute(encoder, this._curlPipeline,    this._buildCurlBindGroup());
      this._dispatchCompute(encoder, this._confinePipeline, this._buildConfinementBindGroup());
      this._swapVelocity();
    }

    this._dispatchCompute(encoder, this._divergencePipeline, this._buildDivergenceBindGroup());

    // Pressure Jacobi iterations (warm-started from previous frame).
    for (let i = 0; i < PHYSICS_DEFAULTS.pressureIterations; i++) {
      this._dispatchCompute(encoder, this._pressurePipeline, this._buildPressureBindGroup());
      this._swapPressure();
    }

    this._dispatchCompute(encoder, this._subtractGradPipeline, this._buildSubtractGradBindGroup());
    this._swapVelocity();

    this._device.queue.submit([encoder.finish()]);
  }

  /** Renders the current velocity field to the canvas. */
  renderToCanvas() {
    this._writeRenderParams();
    const encoder = this._device.createCommandEncoder();
    const renderPass = encoder.beginRenderPass({
      colorAttachments: [{
        view:       this._context.getCurrentTexture().createView(),
        loadOp:     'clear',
        storeOp:    'store',
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
      }],
    });
    renderPass.setPipeline(this._renderPipeline);
    renderPass.setBindGroup(0, this._buildRenderBindGroup());
    renderPass.draw(4);
    renderPass.end();
    this._device.queue.submit([encoder.finish()]);
  }

  /** Injects a Gaussian impulse at position (UV [0,1]) in the given direction. */
  injectImpulse(position, direction, radius, strength) {
    this._writeImpulseParams(position, direction, radius / GRID_SIZE, strength);
    this._submitSingleCompute(this._impulsePipeline, this._buildImpulseBindGroup());
    this._swapVelocity();
  }

  /** Injects a filled disk (spin / explode / implode). Center and radius in UV [0,1]. */
  injectDisk(center, radiusUv, strength, mode) {
    this._writeDiskParams(center, radiusUv, strength, mode);
    this._submitSingleCompute(this._diskPipeline, this._buildDiskBindGroup());
    this._swapVelocity();
  }

  /** Adds per-cell pseudo-random velocity. */
  injectNoise(strength, seed) {
    this._writeNoiseParams(strength, seed);
    this._submitSingleCompute(this._noisePipeline, this._buildNoiseBindGroup());
    this._swapVelocity();
  }

  /** Zeros out velocity and pressure fields (triggered by user pressing Clear). */
  clearVelocity() {
    const bytesPerRowVelocity = GRID_SIZE * 8;  // rg32float: 2 × 4 bytes
    const bytesPerRowPressure = GRID_SIZE * 4;  // r32float: 1 × 4 bytes
    const size = [GRID_SIZE, GRID_SIZE, 1];
    for (const tex of this._velocityTextures) {
      this._device.queue.writeTexture({ texture: tex }, this._velocityZeros, { bytesPerRow: bytesPerRowVelocity }, size);
    }
    for (const tex of this._pressureTextures) {
      this._device.queue.writeTexture({ texture: tex }, this._pressureZeros, { bytesPerRow: bytesPerRowPressure }, size);
    }
  }

  dispose() {
    for (const tex of this._velocityTextures) tex.destroy();
    for (const tex of this._pressureTextures) tex.destroy();
    this._divergenceTexture.destroy();
    this._curlTexture.destroy();
    this._advectParamsBuffer.destroy();
    this._baseParamsBuffer.destroy();
    this._confinementParamsBuffer.destroy();
    this._impulseParamsBuffer.destroy();
    this._diskParamsBuffer.destroy();
    this._noiseParamsBuffer.destroy();
    this._renderParamsBuffer.destroy();
  }

  // ─── Texture creation ────────────────────────────────────────────────────────

  // rg32float: velocity (2 channels). Needs both TEXTURE_BINDING (read) and STORAGE_BINDING (write).
  _createRgTexture() {
    return this._device.createTexture({
      size:   [GRID_SIZE, GRID_SIZE, 1],
      format: 'rg32float',
      usage:  GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_DST,
    });
  }

  // r32float: pressure, divergence, curl (1 channel).
  _createRTexture() {
    return this._device.createTexture({
      size:   [GRID_SIZE, GRID_SIZE, 1],
      format: 'r32float',
      usage:  GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_DST,
    });
  }

  // ─── Pipeline creation ───────────────────────────────────────────────────────

  _buildComputePipeline(shaderSource, entryPoint) {
    const module = this._device.createShaderModule({ code: shaderSource });
    return this._device.createComputePipeline({
      layout:  'auto',
      compute: { module, entryPoint },
    });
  }

  _buildRenderPipeline(shaderSource, format) {
    const module = this._device.createShaderModule({ code: shaderSource });
    return this._device.createRenderPipeline({
      layout:   'auto',
      vertex:   { module, entryPoint: 'vertexMain' },
      fragment: { module, entryPoint: 'fragmentMain', targets: [{ format }] },
      primitive: { topology: 'triangle-strip' },
    });
  }

  // ─── Ping-pong helpers ───────────────────────────────────────────────────────

  _swapVelocity() { this._velocityReadIdx ^= 1; }
  _swapPressure() { this._pressureReadIdx ^= 1; }

  get _velocityRead()  { return this._velocityTextures[this._velocityReadIdx]; }
  get _velocityWrite() { return this._velocityTextures[this._velocityReadIdx ^ 1]; }
  get _pressureRead()  { return this._pressureTextures[this._pressureReadIdx]; }
  get _pressureWrite() { return this._pressureTextures[this._pressureReadIdx ^ 1]; }

  // ─── Dispatch helpers ────────────────────────────────────────────────────────

  _dispatchCompute(encoder, pipeline, bindGroup) {
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(this._workgroupsXY, this._workgroupsXY);
    pass.end();
  }

  _submitSingleCompute(pipeline, bindGroup) {
    const encoder = this._device.createCommandEncoder();
    this._dispatchCompute(encoder, pipeline, bindGroup);
    this._device.queue.submit([encoder.finish()]);
  }

  // ─── Params writing ──────────────────────────────────────────────────────────

  _writeAdvectParams(deltaTime) {
    // Field order must match struct AdvectParams in velocity_advect.wgsl exactly.
    new UniformWriter(16)
      .u32(this._gridSize)
      .u32(PHYSICS_DEFAULTS.boundaryMode)
      .f32(deltaTime)
      .f32(Math.pow(PHYSICS_DEFAULTS.damping, deltaTime))  // per-second damping → per-frame
      .writeBuffer(this._device, this._advectParamsBuffer);
  }

  _writeBaseParams() {
    // Field order must match struct BaseParams in divergence/pressure/subtract_gradient/curl shaders.
    new UniformWriter(16)
      .u32(this._gridSize)
      .u32(PHYSICS_DEFAULTS.boundaryMode)
      .pad()
      .pad()
      .writeBuffer(this._device, this._baseParamsBuffer);
  }

  _writeConfinementParams(deltaTime) {
    // Field order must match struct ConfinementParams in velocity_vorticity_confinement.wgsl.
    new UniformWriter(16)
      .u32(this._gridSize)
      .pad()
      .f32(PHYSICS_DEFAULTS.vorticityStrength)
      .f32(deltaTime)
      .writeBuffer(this._device, this._confinementParamsBuffer);
  }

  _writeImpulseParams(position, direction, radiusUv, strength) {
    // Field order must match struct ImpulseParams in velocity_inject_impulse.wgsl.
    new UniformWriter(32)
      .u32(this._gridSize)
      .u32(PHYSICS_DEFAULTS.boundaryMode)
      .f32(radiusUv)
      .f32(strength)
      .f32(position[0])
      .f32(position[1])
      .f32(direction[0])
      .f32(direction[1])
      .writeBuffer(this._device, this._impulseParamsBuffer);
  }

  _writeDiskParams(center, radiusUv, strength, mode) {
    // Field order must match struct DiskParams in velocity_inject_disk.wgsl.
    new UniformWriter(32)
      .u32(this._gridSize)
      .u32(PHYSICS_DEFAULTS.boundaryMode)
      .f32(radiusUv)
      .f32(strength)
      .f32(center[0])
      .f32(center[1])
      .u32(mode)
      .pad()
      .writeBuffer(this._device, this._diskParamsBuffer);
  }

  _writeNoiseParams(strength, seed) {
    // Field order must match struct NoiseParams in velocity_noise.wgsl.
    new UniformWriter(16)
      .u32(this._gridSize)
      .pad()
      .f32(seed)
      .f32(strength)
      .writeBuffer(this._device, this._noiseParamsBuffer);
  }

  _writeRenderParams() {
    // Field order must match struct RenderParams in velocity_render.wgsl.
    new UniformWriter(16)
      .u32(this._gridSize)
      .pad()
      .f32(RENDER_DEFAULTS.velocityToneMidpoint)
      .pad()
      .writeBuffer(this._device, this._renderParamsBuffer);
  }

  // ─── Bind group builders ─────────────────────────────────────────────────────

  _buildAdvectBindGroup() {
    return this._device.createBindGroup({
      layout:  this._advectPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this._velocityRead.createView() },
        { binding: 1, resource: this._velocityWrite.createView() },
        { binding: 2, resource: { buffer: this._advectParamsBuffer } },
      ],
    });
  }

  _buildDivergenceBindGroup() {
    return this._device.createBindGroup({
      layout:  this._divergencePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this._velocityRead.createView() },
        { binding: 1, resource: this._divergenceTexture.createView() },
        { binding: 2, resource: { buffer: this._baseParamsBuffer } },
      ],
    });
  }

  _buildPressureBindGroup() {
    return this._device.createBindGroup({
      layout:  this._pressurePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this._pressureRead.createView() },
        { binding: 1, resource: this._divergenceTexture.createView() },
        { binding: 2, resource: this._pressureWrite.createView() },
        { binding: 3, resource: { buffer: this._baseParamsBuffer } },
      ],
    });
  }

  _buildSubtractGradBindGroup() {
    return this._device.createBindGroup({
      layout:  this._subtractGradPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this._velocityRead.createView() },
        { binding: 1, resource: this._pressureRead.createView() },
        { binding: 2, resource: this._velocityWrite.createView() },
        { binding: 3, resource: { buffer: this._baseParamsBuffer } },
      ],
    });
  }

  _buildCurlBindGroup() {
    return this._device.createBindGroup({
      layout:  this._curlPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this._velocityRead.createView() },
        { binding: 1, resource: this._curlTexture.createView() },
        { binding: 2, resource: { buffer: this._baseParamsBuffer } },
      ],
    });
  }

  _buildConfinementBindGroup() {
    return this._device.createBindGroup({
      layout:  this._confinePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this._velocityRead.createView() },
        { binding: 1, resource: this._curlTexture.createView() },
        { binding: 2, resource: this._velocityWrite.createView() },
        { binding: 3, resource: { buffer: this._confinementParamsBuffer } },
      ],
    });
  }

  _buildImpulseBindGroup() {
    return this._device.createBindGroup({
      layout:  this._impulsePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this._velocityRead.createView() },
        { binding: 1, resource: this._velocityWrite.createView() },
        { binding: 2, resource: { buffer: this._impulseParamsBuffer } },
      ],
    });
  }

  _buildDiskBindGroup() {
    return this._device.createBindGroup({
      layout:  this._diskPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this._velocityRead.createView() },
        { binding: 1, resource: this._velocityWrite.createView() },
        { binding: 2, resource: { buffer: this._diskParamsBuffer } },
      ],
    });
  }

  _buildNoiseBindGroup() {
    return this._device.createBindGroup({
      layout:  this._noisePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this._velocityRead.createView() },
        { binding: 1, resource: this._velocityWrite.createView() },
        { binding: 2, resource: { buffer: this._noiseParamsBuffer } },
      ],
    });
  }

  _buildRenderBindGroup() {
    return this._device.createBindGroup({
      layout:  this._renderPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this._velocityRead.createView() },
        { binding: 1, resource: { buffer: this._renderParamsBuffer } },
      ],
    });
  }

  // ─── Initialisation ──────────────────────────────────────────────────────────

  _clearAllTextures() {
    const bytesPerRowV = GRID_SIZE * 8;
    const bytesPerRowP = GRID_SIZE * 4;
    const size = [GRID_SIZE, GRID_SIZE, 1];
    for (const tex of this._velocityTextures) {
      this._device.queue.writeTexture({ texture: tex }, this._velocityZeros, { bytesPerRow: bytesPerRowV }, size);
    }
    for (const tex of this._pressureTextures) {
      this._device.queue.writeTexture({ texture: tex }, this._pressureZeros, { bytesPerRow: bytesPerRowP }, size);
    }
    this._device.queue.writeTexture({ texture: this._divergenceTexture }, this._pressureZeros, { bytesPerRow: bytesPerRowP }, size);
    this._device.queue.writeTexture({ texture: this._curlTexture },       this._pressureZeros, { bytesPerRow: bytesPerRowP }, size);
  }
}

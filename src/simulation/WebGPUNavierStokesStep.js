'use strict';

import {
  GRID_SIZE, PHYSICS_DEFAULTS, RENDER_DEFAULTS, GRAVITY,
} from '../config/SimulationConfig.js';
import { UniformWriter } from '../gpu/UniformWriter.js';

const WORKGROUP_SIZE = 8; // threads per dimension; dispatch is ceil(N/8) × ceil(N/8)

// Jacobi pressure steps run inside one dispatch on a workgroup-memory tile. More steps per
// dispatch mean fewer dispatches but a thicker tile border of redundant arithmetic.
const PRESSURE_ITERATIONS_PER_DISPATCH = 4;
const PRESSURE_TILE_SIZE               = 32;  // must match TILE_SIZE in velocity_pressure.wgsl

// Noise blobs sit this fraction of their radius apart, so neighbours overlap and leave no gaps.
const NOISE_LATTICE_SPACING_FRACTION = 0.75;

/**
 * Physics step implementation: incompressible Navier-Stokes, fully on WebGPU.
 *
 * Interface contract (all physics steps must implement):
 *   step(deltaTime)
 *   injectImpulse(position, direction, radius, strength)
 *   injectDisk(center, radiusUv, strength, mode)
 *   injectNoise(strength, seed, blobRadius)
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
    this._bindGroupCache    = new Map();

    // Ping-pong textures for density field and gravity potential (r32float).
    this._densityTextures          = [this._createRTexture(), this._createRTexture()];
    this._gravityPotentialTextures = [this._createRTexture(), this._createRTexture()];
    this._densityReadIdx          = 0;
    this._gravityPotentialReadIdx = 0;

    // Zero-filled arrays for clearing textures (allocated once, reused).
    this._velocityZeros = new Float32Array(GRID_SIZE * GRID_SIZE * 2);
    this._pressureZeros = new Float32Array(GRID_SIZE * GRID_SIZE);
    this._densityZeros  = new Float32Array(GRID_SIZE * GRID_SIZE);

    // Uniform buffers — velocity field.
    this._advectParamsBuffer      = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this._baseParamsBuffer        = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    // Full pressure dispatches and the shorter last one need different iteration counts
    // within one pass, so each has its own uniform buffer.
    this._pressureBlockParamsBuffer     = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this._pressureRemainderParamsBuffer = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this._confinementParamsBuffer = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this._impulseParamsBuffer     = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this._diskParamsBuffer        = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this._noiseParamsBuffer       = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this._renderParamsBuffer      = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

    // Uniform buffers — density field and gravity.
    this._densityAdvectParamsBuffer  = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this._gravityPoissonParamsBuffer = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this._gravityForceParamsBuffer   = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this._densityImpulseParamsBuffer = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this._densityDiskParamsBuffer    = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this._densityNoiseParamsBuffer   = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

    // Build compute and render pipelines — velocity field.
    this._advectPipeline       = this._buildComputePipeline(shaderSources.advect,               'advectVelocity');
    this._divergencePipeline   = this._buildComputePipeline(shaderSources.divergence,           'computeDivergence');
    this._pressurePipeline     = this._buildComputePipeline(shaderSources.pressure,             'jacobiPressureBlock');
    this._subtractGradPipeline = this._buildComputePipeline(shaderSources.subtractGradient,     'subtractPressureGradient');
    this._curlPipeline         = this._buildComputePipeline(shaderSources.vorticityCurl,        'computeCurl');
    this._confinePipeline      = this._buildComputePipeline(shaderSources.vorticityConfinement, 'applyVorticityConfinement');
    this._impulsePipeline      = this._buildComputePipeline(shaderSources.injectImpulse,        'injectGaussianImpulse');
    this._diskPipeline         = this._buildComputePipeline(shaderSources.injectDisk,           'injectDisk');
    this._noisePipeline        = this._buildComputePipeline(shaderSources.noise,                'injectNoise');
    this._renderPipeline       = this._buildRenderPipeline(shaderSources.render, this._canvasFormat);

    // Build compute pipelines — density field and gravity.
    this._densityAdvectPipeline  = this._buildComputePipeline(shaderSources.densityAdvect,        'advectDensity');
    this._gravityPoissonPipeline = this._buildComputePipeline(shaderSources.gravityPoisson,       'jacobiGravityStep');
    this._gravityForcePipeline   = this._buildComputePipeline(shaderSources.gravityForce,         'applyGravityForce');
    this._densityImpulsePipeline = this._buildComputePipeline(shaderSources.densityInjectImpulse, 'injectDensityImpulse');
    this._densityDiskPipeline    = this._buildComputePipeline(shaderSources.densityInjectDisk,    'injectDensityDisk');
    this._densityNoisePipeline   = this._buildComputePipeline(shaderSources.densityInjectNoise,   'injectDensityNoise');

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

    // One compute pass for the whole step: WebGPU orders the dispatches of a pass and makes
    // each one see the previous one's writes, while every extra pass costs GPU idle time.
    const encoder = this._device.createCommandEncoder();
    const pass    = encoder.beginComputePass();

    this._dispatch(pass, this._advectPipeline, this._cachedBindGroup(this._buildAdvectBindGroup));
    this._swapVelocity();

    if (PHYSICS_DEFAULTS.vorticityStrength > 0) {
      this._writeConfinementParams(deltaTime);
      this._dispatch(pass, this._curlPipeline,    this._cachedBindGroup(this._buildCurlBindGroup));
      this._dispatch(pass, this._confinePipeline, this._cachedBindGroup(this._buildConfinementBindGroup));
      this._swapVelocity();
    }

    if (GRAVITY.enabled) {
      this._writeDensityAdvectParams(deltaTime);
      this._dispatch(pass, this._densityAdvectPipeline, this._cachedBindGroup(this._buildDensityAdvectBindGroup));
      this._swapDensity();

      // Gravity potential Jacobi iterations (warm-started from previous frame).
      this._writeGravityPoissonParams();
      for (let i = 0; i < GRAVITY.iterations; i++) {
        this._dispatch(pass, this._gravityPoissonPipeline, this._cachedBindGroup(this._buildGravityPoissonBindGroup));
        this._swapGravityPotential();
      }

      this._writeGravityForceParams(deltaTime);
      this._dispatch(pass, this._gravityForcePipeline, this._cachedBindGroup(this._buildGravityForceBindGroup));
      this._swapVelocity();
    }

    this._dispatch(pass, this._divergencePipeline, this._cachedBindGroup(this._buildDivergenceBindGroup));

    this._dispatchPressureIterations(pass, PHYSICS_DEFAULTS.pressureIterations);

    this._dispatch(pass, this._subtractGradPipeline, this._cachedBindGroup(this._buildSubtractGradBindGroup));
    this._swapVelocity();

    pass.end();
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
    renderPass.setBindGroup(0, this._cachedBindGroup(this._buildRenderBindGroup));
    renderPass.draw(4);
    renderPass.end();
    this._device.queue.submit([encoder.finish()]);
  }

  /** Injects a Gaussian impulse at position (UV [0,1]) in the given direction. */
  injectImpulse(position, direction, radius, strength) {
    this._writeImpulseParams(position, direction, radius / GRID_SIZE, strength);
    this._submitSingleCompute(this._impulsePipeline, this._cachedBindGroup(this._buildImpulseBindGroup));
    this._swapVelocity();
  }

  /** Injects a filled disk (spin / explode / implode). Center and radius in UV [0,1]. */
  injectDisk(center, radiusUv, strength, mode) {
    this._writeDiskParams(center, radiusUv, strength, mode);
    this._submitSingleCompute(this._diskPipeline, this._cachedBindGroup(this._buildDiskBindGroup));
    this._swapVelocity();
  }

  /**
   * Adds random velocity: a random direction per cell when blobRadius (grid cells) ≤ 1,
   * otherwise overlapping Gaussian blobs of that radius with random directions. seed is a u32.
   */
  injectNoise(strength, seed, blobRadius) {
    this._writeNoiseParams(this._noiseParamsBuffer, strength, seed, blobRadius);
    this._submitSingleCompute(this._noisePipeline, this._cachedBindGroup(this._buildNoiseBindGroup));
    this._swapVelocity();
  }

  /** Zeros out velocity and pressure fields. */
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

  /** Zeros out the density field and gravity potential. */
  clearDensity() {
    const bytesPerRow = GRID_SIZE * 4;
    const size = [GRID_SIZE, GRID_SIZE, 1];
    for (const tex of this._densityTextures) {
      this._device.queue.writeTexture({ texture: tex }, this._densityZeros, { bytesPerRow }, size);
    }
    // Clear potential too — stale potential would cause spurious gravity forces after clear.
    for (const tex of this._gravityPotentialTextures) {
      this._device.queue.writeTexture({ texture: tex }, this._densityZeros, { bytesPerRow }, size);
    }
  }

  /** Injects a Gaussian density blob at position (UV [0,1]). */
  injectDensityImpulse(position, radius, strength) {
    this._writeDensityImpulseParams(position, radius / GRID_SIZE, strength);
    this._submitSingleCompute(this._densityImpulsePipeline, this._cachedBindGroup(this._buildDensityImpulseBindGroup));
    this._swapDensity();
  }

  /** Injects a filled disk of density. Center and radius in UV [0,1]. */
  injectDensityDisk(center, radiusUv, strength) {
    this._writeDensityDiskParams(center, radiusUv, strength);
    this._submitSingleCompute(this._densityDiskPipeline, this._cachedBindGroup(this._buildDensityDiskBindGroup));
    this._swapDensity();
  }

  /** Adds random density, per cell or in blobs, like injectNoise. */
  injectDensityNoise(strength, seed, blobRadius) {
    this._writeNoiseParams(this._densityNoiseParamsBuffer, strength, seed, blobRadius);
    this._submitSingleCompute(this._densityNoisePipeline, this._cachedBindGroup(this._buildDensityNoiseBindGroup));
    this._swapDensity();
  }

  dispose() {
    for (const tex of this._velocityTextures) tex.destroy();
    for (const tex of this._pressureTextures) tex.destroy();
    for (const tex of this._densityTextures) tex.destroy();
    for (const tex of this._gravityPotentialTextures) tex.destroy();
    this._divergenceTexture.destroy();
    this._curlTexture.destroy();
    this._advectParamsBuffer.destroy();
    this._baseParamsBuffer.destroy();
    this._pressureBlockParamsBuffer.destroy();
    this._pressureRemainderParamsBuffer.destroy();
    this._confinementParamsBuffer.destroy();
    this._impulseParamsBuffer.destroy();
    this._diskParamsBuffer.destroy();
    this._noiseParamsBuffer.destroy();
    this._renderParamsBuffer.destroy();
    this._densityAdvectParamsBuffer.destroy();
    this._gravityPoissonParamsBuffer.destroy();
    this._gravityForceParamsBuffer.destroy();
    this._densityImpulseParamsBuffer.destroy();
    this._densityDiskParamsBuffer.destroy();
    this._densityNoiseParamsBuffer.destroy();
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

  _swapVelocity()         { this._velocityReadIdx ^= 1; }
  _swapPressure()         { this._pressureReadIdx ^= 1; }
  _swapDensity()          { this._densityReadIdx ^= 1; }
  _swapGravityPotential() { this._gravityPotentialReadIdx ^= 1; }

  get _velocityRead()          { return this._velocityTextures[this._velocityReadIdx]; }
  get _velocityWrite()         { return this._velocityTextures[this._velocityReadIdx ^ 1]; }
  get _pressureRead()          { return this._pressureTextures[this._pressureReadIdx]; }
  get _pressureWrite()         { return this._pressureTextures[this._pressureReadIdx ^ 1]; }
  get _densityRead()           { return this._densityTextures[this._densityReadIdx]; }
  get _densityWrite()          { return this._densityTextures[this._densityReadIdx ^ 1]; }
  get _gravityPotentialRead()  { return this._gravityPotentialTextures[this._gravityPotentialReadIdx]; }
  get _gravityPotentialWrite() { return this._gravityPotentialTextures[this._gravityPotentialReadIdx ^ 1]; }

  // ─── Dispatch helpers ────────────────────────────────────────────────────────

  // Jacobi pressure iterations, warm-started from the previous frame's solution.
  _dispatchPressureIterations(pass, iterationCount) {
    const fullDispatches      = Math.floor(iterationCount / PRESSURE_ITERATIONS_PER_DISPATCH);
    const remainderIterations = iterationCount % PRESSURE_ITERATIONS_PER_DISPATCH;
    this._writePressureParams(this._pressureBlockParamsBuffer, PRESSURE_ITERATIONS_PER_DISPATCH);
    this._writePressureParams(this._pressureRemainderParamsBuffer, remainderIterations);

    const blocks = this._pressureBlocksPerAxis(PRESSURE_ITERATIONS_PER_DISPATCH);
    for (let i = 0; i < fullDispatches; i++) {
      pass.setPipeline(this._pressurePipeline);
      pass.setBindGroup(0, this._cachedBindGroup(this._buildPressureBlockBindGroup));
      pass.dispatchWorkgroups(blocks, blocks);
      this._swapPressure();
    }
    if (remainderIterations > 0) {
      const remainderBlocks = this._pressureBlocksPerAxis(remainderIterations);
      pass.setPipeline(this._pressurePipeline);
      pass.setBindGroup(0, this._cachedBindGroup(this._buildPressureRemainderBindGroup));
      pass.dispatchWorkgroups(remainderBlocks, remainderBlocks);
      this._swapPressure();
    }
  }

  // Each workgroup writes the tile minus a border as wide as its iteration count.
  _pressureBlocksPerAxis(iterationsPerDispatch) {
    return Math.ceil(this._gridSize / (PRESSURE_TILE_SIZE - 2 * iterationsPerDispatch));
  }

  _dispatch(pass, pipeline, bindGroup) {
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(this._workgroupsXY, this._workgroupsXY);
  }

  _submitSingleCompute(pipeline, bindGroup) {
    const encoder = this._device.createCommandEncoder();
    const pass    = encoder.beginComputePass();
    this._dispatch(pass, pipeline, bindGroup);
    pass.end();
    this._device.queue.submit([encoder.finish()]);
  }

  // Every bind group depends only on which texture of each ping-pong pair is current,
  // so a builder runs once per combination of read indices and its result is reused.
  _cachedBindGroup(builder) {
    const pingPongState = `${this._velocityReadIdx}${this._pressureReadIdx}${this._densityReadIdx}${this._gravityPotentialReadIdx}`;
    let byState = this._bindGroupCache.get(builder);
    if (!byState) {
      byState = new Map();
      this._bindGroupCache.set(builder, byState);
    }
    let bindGroup = byState.get(pingPongState);
    if (!bindGroup) {
      bindGroup = builder.call(this);
      byState.set(pingPongState, bindGroup);
    }
    return bindGroup;
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

  _writePressureParams(buffer, iterationsPerDispatch) {
    // Field order must match struct PressureParams in velocity_pressure.wgsl.
    new UniformWriter(16)
      .u32(this._gridSize)
      .u32(PHYSICS_DEFAULTS.boundaryMode)
      .u32(iterationsPerDispatch)
      .pad()
      .writeBuffer(this._device, buffer);
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

  _writeNoiseParams(paramsBuffer, strength, seed, blobRadius) {
    const latticeNodesPerSide = Math.max(1, Math.ceil(this._gridSize / (NOISE_LATTICE_SPACING_FRACTION * blobRadius)));
    // Field order must match struct NoiseParams in noise_lattice.wgsl.
    new UniformWriter(32)
      .u32(this._gridSize)
      .u32(PHYSICS_DEFAULTS.boundaryMode)
      .u32(seed)
      .u32(latticeNodesPerSide)
      .f32(strength)
      .f32(blobRadius)
      .pad()
      .pad()
      .writeBuffer(this._device, paramsBuffer);
  }

  _writeRenderParams() {
    // Field order must match struct RenderParams in velocity_render.wgsl.
    new UniformWriter(16)
      .u32(this._gridSize)
      .pad()
      .f32(RENDER_DEFAULTS.velocityToneMidpoint)
      .f32(RENDER_DEFAULTS.densityToneMidpoint)
      .writeBuffer(this._device, this._renderParamsBuffer);
  }

  _writeDensityAdvectParams(deltaTime) {
    // Field order must match struct DensityAdvectParams in density_advect.wgsl.
    new UniformWriter(32)
      .u32(this._gridSize)
      .u32(PHYSICS_DEFAULTS.boundaryMode)
      .f32(deltaTime)
      .f32(Math.pow(GRAVITY.densityDamping, deltaTime))
      .f32(GRAVITY.strength)
      .pad()
      .pad()
      .pad()
      .writeBuffer(this._device, this._densityAdvectParamsBuffer);
  }

  _writeGravityPoissonParams() {
    // Field order must match struct GravityPoissonParams in gravity_poisson.wgsl.
    new UniformWriter(16)
      .u32(this._gridSize)
      .u32(PHYSICS_DEFAULTS.boundaryMode)
      .pad()
      .pad()
      .writeBuffer(this._device, this._gravityPoissonParamsBuffer);
  }

  _writeGravityForceParams(deltaTime) {
    // Field order must match struct GravityForceParams in gravity_force.wgsl.
    new UniformWriter(16)
      .u32(this._gridSize)
      .u32(PHYSICS_DEFAULTS.boundaryMode)
      .f32(GRAVITY.strength)
      .f32(deltaTime)
      .writeBuffer(this._device, this._gravityForceParamsBuffer);
  }

  _writeDensityImpulseParams(position, radiusUv, strength) {
    // Field order must match struct DensityImpulseParams in density_inject_impulse.wgsl.
    new UniformWriter(32)
      .u32(this._gridSize)
      .u32(PHYSICS_DEFAULTS.boundaryMode)
      .f32(radiusUv)
      .f32(strength)
      .f32(position[0])
      .f32(position[1])
      .pad()
      .pad()
      .writeBuffer(this._device, this._densityImpulseParamsBuffer);
  }

  _writeDensityDiskParams(center, radiusUv, strength) {
    // Field order must match struct DensityDiskParams in density_inject_disk.wgsl.
    new UniformWriter(32)
      .u32(this._gridSize)
      .u32(PHYSICS_DEFAULTS.boundaryMode)
      .f32(radiusUv)
      .f32(strength)
      .f32(center[0])
      .f32(center[1])
      .pad()
      .pad()
      .writeBuffer(this._device, this._densityDiskParamsBuffer);
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

  _buildPressureBlockBindGroup() {
    return this._buildPressureBindGroup(this._pressureBlockParamsBuffer);
  }

  _buildPressureRemainderBindGroup() {
    return this._buildPressureBindGroup(this._pressureRemainderParamsBuffer);
  }

  _buildPressureBindGroup(paramsBuffer) {
    return this._device.createBindGroup({
      layout:  this._pressurePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this._pressureRead.createView() },
        { binding: 1, resource: this._divergenceTexture.createView() },
        { binding: 2, resource: this._pressureWrite.createView() },
        { binding: 3, resource: { buffer: paramsBuffer } },
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
        { binding: 2, resource: this._densityRead.createView() },
      ],
    });
  }

  _buildDensityAdvectBindGroup() {
    return this._device.createBindGroup({
      layout:  this._densityAdvectPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this._velocityRead.createView() },
        { binding: 1, resource: this._densityRead.createView() },
        { binding: 2, resource: this._densityWrite.createView() },
        { binding: 3, resource: { buffer: this._densityAdvectParamsBuffer } },
        { binding: 4, resource: this._gravityPotentialRead.createView() },
      ],
    });
  }

  _buildGravityPoissonBindGroup() {
    return this._device.createBindGroup({
      layout:  this._gravityPoissonPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this._gravityPotentialRead.createView() },
        { binding: 1, resource: this._densityRead.createView() },
        { binding: 2, resource: this._gravityPotentialWrite.createView() },
        { binding: 3, resource: { buffer: this._gravityPoissonParamsBuffer } },
      ],
    });
  }

  _buildGravityForceBindGroup() {
    return this._device.createBindGroup({
      layout:  this._gravityForcePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this._velocityRead.createView() },
        { binding: 1, resource: this._gravityPotentialRead.createView() },
        { binding: 2, resource: this._velocityWrite.createView() },
        { binding: 3, resource: { buffer: this._gravityForceParamsBuffer } },
      ],
    });
  }

  _buildDensityImpulseBindGroup() {
    return this._device.createBindGroup({
      layout:  this._densityImpulsePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this._densityRead.createView() },
        { binding: 1, resource: this._densityWrite.createView() },
        { binding: 2, resource: { buffer: this._densityImpulseParamsBuffer } },
      ],
    });
  }

  _buildDensityDiskBindGroup() {
    return this._device.createBindGroup({
      layout:  this._densityDiskPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this._densityRead.createView() },
        { binding: 1, resource: this._densityWrite.createView() },
        { binding: 2, resource: { buffer: this._densityDiskParamsBuffer } },
      ],
    });
  }

  _buildDensityNoiseBindGroup() {
    return this._device.createBindGroup({
      layout:  this._densityNoisePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this._densityRead.createView() },
        { binding: 1, resource: this._densityWrite.createView() },
        { binding: 2, resource: { buffer: this._densityNoiseParamsBuffer } },
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
    for (const tex of this._densityTextures) {
      this._device.queue.writeTexture({ texture: tex }, this._densityZeros, { bytesPerRow: bytesPerRowP }, size);
    }
    for (const tex of this._gravityPotentialTextures) {
      this._device.queue.writeTexture({ texture: tex }, this._densityZeros, { bytesPerRow: bytesPerRowP }, size);
    }
    this._device.queue.writeTexture({ texture: this._divergenceTexture }, this._pressureZeros, { bytesPerRow: bytesPerRowP }, size);
    this._device.queue.writeTexture({ texture: this._curlTexture },       this._pressureZeros, { bytesPerRow: bytesPerRowP }, size);
  }
}

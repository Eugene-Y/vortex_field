'use strict';

import { GRID_SIZE, ROTATION_FIELD, ROTATION_REFERENCE_GRID_SIZE, RENDER_DEFAULTS, COLORS, PHYSICS_DEFAULTS } from '../config/SimulationConfig.js';
import { UniformWriter } from '../gpu/UniformWriter.js';

const WORKGROUP_SIZE = 64;
const TILE_WIDTH     = 8;  // must match TILE_WIDTH in rotation_compute.wgsl

// Resolution of the 64-bit fixed-point Field B sums: 2⁻³² of a raw ω unit, range ±2³¹ units.
const FIXED_POINT_SCALE           = 2 ** 32;
const ACCUMULATION_WORDS_PER_CELL = 2;

// A focus-mask bounding box has too few cells to occupy the GPU on its own,
// so its span table is split over this many workgroup rows.
const MASK_SPAN_SPLIT = 16;

// [dRow, dColStart, dColEnd, unused] — matches vec4<i32> in rotation_compute.wgsl.
const SPAN_INTS = 4;

/**
 * Field B — WebGPU implementation.
 *
 * Three passes per frame:
 *   1. Compute pass: one thread per cell, each walks the precomputed half-plane annulus
 *      span table and accumulates signed ω into 64-bit fixed-point sums.
 *   2. Reduce pass: fixed-point sums → flat f32 output buffer.
 *   3. Render pass: output buffer → canvas-rotation.
 *
 * External interface:
 *   recomputeFrom(velocityGpuTexture)  — takes current velocity GPUTexture directly
 *   dispose()
 */
export class WebGPURotationField {
  constructor(device, canvas, computeShaderSource, reduceShaderSource, energyShaderSource, renderShaderSource) {
    this._device   = device;
    this._canvas   = canvas;
    this._gridSize = GRID_SIZE;

    const totalCells = GRID_SIZE * GRID_SIZE;

    this._context = canvas.getContext('webgpu');
    const format  = navigator.gpu.getPreferredCanvasFormat();
    this._context.configure({ device, format, alphaMode: 'opaque' });

    // Two u32 words per cell (64-bit fixed point) — cleared each frame.
    this._atomicBuffer = device.createBuffer({
      size:  ACCUMULATION_WORDS_PER_CELL * totalCells * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // Flat f32 buffer: reduction output, also used directly by render shader.
    this._outputBuffer = device.createBuffer({
      size:  totalCells * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // Single f32: EMA of Σ|output[i]|, updated each frame by the energy pass.
    // Zero-initialized by WebGPU; cold-start logic in the shader handles first frame.
    this._smoothedEnergyBuffer = device.createBuffer({
      size:  4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    this._energyParamsBuffer = device.createBuffer({
      size:  16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this._computeParamsBuffer = device.createBuffer({
      size:  64,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this._reduceParamsBuffer = device.createBuffer({
      size:  16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this._renderParamsBuffer = device.createBuffer({
      size:  64,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Precomputed half-plane annulus spans. Each row dRow ∈ [0, gridSize/2] holds at most
    // two runs (left and right of the hole), which bounds the table size.
    this._annulusSpanBuffer = device.createBuffer({
      size:  (GRID_SIZE + 2) * SPAN_INTS * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this._annulusSpanCount = 0;
    this._annulusMinRadius = 0;
    // Sentinel values that force a rebuild on the first frame.
    this._lastPairDistance  = -1;
    this._lastDistanceDelta = -1;

    // Output cache: tracks whether the output buffer reflects current velocity + compute params.
    // Allows skipping compute+reduce when only render params (brightness) change.
    this._lastVelocityGeneration   = -1;
    this._lastComputeParamSnapshot = null;
    this._hasValidOutput           = false;

    // Compute bind group cache keyed by velocity GPUTexture.
    // Ping-pong means at most 2 entries. Cleared when the span buffer is rebuilt.
    this._computeBindGroupCache = new Map();

    this._computePipeline = this._buildComputePipeline(computeShaderSource);
    this._reducePipeline  = this._buildReducePipeline(reduceShaderSource);
    this._energyPipeline  = this._buildEnergyPipeline(energyShaderSource);
    this._renderPipeline  = this._buildRenderPipeline(renderShaderSource, format);

    // Reduce, energy, and render bind groups are stable (don't depend on the velocity texture).
    this._reduceBindGroup = this._buildReduceBindGroup();
    this._energyBindGroup = this._buildEnergyBindGroup();
    this._renderBindGroup = this._buildRenderBindGroup();

    this._writeReduceParams();
    this._writeEnergyParams();
    this._writeRenderParams();
    this._lastRenderParamSnapshot = this._snapshotRenderParams();
  }

  /**
   * Runs compute+reduce+render passes using the supplied velocity GPUTexture.
   * Skips compute+reduce and re-renders from the cached output buffer when:
   *   - the velocity texture hasn't changed (simulation is paused), AND
   *   - no compute-relevant params changed (only brightness was adjusted).
   * Called once per frame from the main render loop.
   */
  recomputeFrom(velocityGpuTexture, velocityGeneration) {
    const velocityChanged      = velocityGeneration !== this._lastVelocityGeneration;
    const computeParamsChanged = this._haveComputeParamsChanged();

    if (this._haveRenderParamsChanged()) {
      this._writeRenderParams();
      this._lastRenderParamSnapshot = this._snapshotRenderParams();
    }

    if (!velocityChanged && !computeParamsChanged && this._hasValidOutput) {
      this._submitRenderPass();
      return;
    }

    this._writeComputeParams();
    this._lastVelocityGeneration   = velocityGeneration;
    this._lastComputeParamSnapshot = this._snapshotComputeParams();
    this._hasValidOutput           = true;

    const totalCells = this._gridSize * this._gridSize;
    const encoder    = this._device.createCommandEncoder();

    // Clear the accumulation and output buffers before accumulation.
    encoder.clearBuffer(this._atomicBuffer);
    encoder.clearBuffer(this._outputBuffer);

    // Compute pass: accumulate rotation contributions.
    // Rebuild bind group each frame — velocity texture ping-pong changes which texture is current.
    const computeBindGroup = this._buildComputeBindGroup(velocityGpuTexture);
    const computePass = encoder.beginComputePass();
    computePass.setPipeline(this._computePipeline);
    computePass.setBindGroup(0, computeBindGroup);
    computePass.dispatchWorkgroups(...this._computeDispatchSize());
    computePass.end();

    // Reduce pass: fixed-point sums → output buffer.
    const reducePass = encoder.beginComputePass();
    reducePass.setPipeline(this._reducePipeline);
    reducePass.setBindGroup(0, this._reduceBindGroup);
    reducePass.dispatchWorkgroups(Math.ceil(totalCells / WORKGROUP_SIZE));
    reducePass.end();

    // Energy pass: update smoothed mean absolute energy for auto-brightness.
    // Skipped when auto-normalization is off — no cost to the disabled feature.
    if (ROTATION_FIELD.autoNormalize) {
      const energyPass = encoder.beginComputePass();
      energyPass.setPipeline(this._energyPipeline);
      energyPass.setBindGroup(0, this._energyBindGroup);
      energyPass.dispatchWorkgroups(1);
      energyPass.end();
    }

    this._submitRenderPassInto(encoder);
    this._device.queue.submit([encoder.finish()]);
  }

  _submitRenderPass() {
    const encoder = this._device.createCommandEncoder();
    this._submitRenderPassInto(encoder);
    this._device.queue.submit([encoder.finish()]);
  }

  _submitRenderPassInto(encoder) {
    const renderPass = encoder.beginRenderPass({
      colorAttachments: [{
        view:       this._context.getCurrentTexture().createView(),
        loadOp:     'clear',
        storeOp:    'store',
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
      }],
    });
    renderPass.setPipeline(this._renderPipeline);
    renderPass.setBindGroup(0, this._renderBindGroup);
    renderPass.draw(4);
    renderPass.end();
  }

  dispose() {
    this._atomicBuffer.destroy();
    this._outputBuffer.destroy();
    this._smoothedEnergyBuffer.destroy();
    this._computeParamsBuffer.destroy();
    this._reduceParamsBuffer.destroy();
    this._energyParamsBuffer.destroy();
    this._renderParamsBuffer.destroy();
    this._annulusSpanBuffer.destroy();
  }

  _buildComputePipeline(shaderSource) {
    const module = this._device.createShaderModule({ code: shaderSource });
    return this._device.createComputePipeline({
      layout: 'auto',
      compute: {
        module,
        entryPoint: 'computeRotation',
        constants:  { FIXED_POINT_SCALE },
      },
    });
  }

  _buildReducePipeline(shaderSource) {
    const module = this._device.createShaderModule({ code: shaderSource });
    return this._device.createComputePipeline({
      layout: 'auto',
      compute: { module, entryPoint: 'reduceBuffers', constants: { FIXED_POINT_SCALE } },
    });
  }

  _buildEnergyPipeline(shaderSource) {
    const module = this._device.createShaderModule({ code: shaderSource });
    return this._device.createComputePipeline({
      layout: 'auto',
      compute: { module, entryPoint: 'computeSmoothedEnergy' },
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

  _buildComputeBindGroup(velocityTexture) {
    let bindGroup = this._computeBindGroupCache.get(velocityTexture);
    if (!bindGroup) {
      bindGroup = this._device.createBindGroup({
        layout: this._computePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: velocityTexture.createView() },
          { binding: 1, resource: { buffer: this._atomicBuffer } },
          { binding: 2, resource: { buffer: this._computeParamsBuffer } },
          { binding: 3, resource: { buffer: this._annulusSpanBuffer } },
        ],
      });
      this._computeBindGroupCache.set(velocityTexture, bindGroup);
    }
    return bindGroup;
  }

  _buildReduceBindGroup() {
    return this._device.createBindGroup({
      layout: this._reducePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this._atomicBuffer } },
        { binding: 1, resource: { buffer: this._outputBuffer } },
        { binding: 2, resource: { buffer: this._reduceParamsBuffer } },
      ],
    });
  }

  _buildEnergyBindGroup() {
    return this._device.createBindGroup({
      layout: this._energyPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this._outputBuffer } },
        { binding: 1, resource: { buffer: this._smoothedEnergyBuffer } },
        { binding: 2, resource: { buffer: this._energyParamsBuffer } },
      ],
    });
  }

  _buildRenderBindGroup() {
    return this._device.createBindGroup({
      layout: this._renderPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this._outputBuffer } },
        { binding: 1, resource: { buffer: this._renderParamsBuffer } },
        { binding: 2, resource: { buffer: this._smoothedEnergyBuffer } },
      ],
    });
  }

  // Without a mask: one workgroup per TILE_WIDTH² tile of cells.
  // With a mask: the bounding-box cells only (O(R²) vs O(N²)), times MASK_SPAN_SPLIT.
  _computeDispatchSize() {
    const mask = this._resolveMaskBox();
    if (mask.active) return [Math.ceil(mask.boxSize * mask.boxSize / WORKGROUP_SIZE), MASK_SPAN_SPLIT];
    const tilesPerSide = Math.ceil(this._gridSize / TILE_WIDTH);
    return [tilesPerSide * tilesPerSide, 1];
  }

  // Returns a plain-object snapshot of all params that feed into the compute pass.
  // Used to detect whether recompute is needed vs render-only (brightness change).
  _snapshotComputeParams() {
    return {
      pairDistance:  ROTATION_FIELD.pairDistance,
      distanceDelta: ROTATION_FIELD.distanceDelta,
      sampleStride:  ROTATION_FIELD.sampleStride,
      boundaryMode:  PHYSICS_DEFAULTS.boundaryMode,
      showMask:      ROTATION_FIELD.showMask,
      maskCenter:    ROTATION_FIELD.maskCenter,
      maskRadius:    ROTATION_FIELD.maskRadius,
    };
  }

  // Colors are compared by reference: the pickers replace the arrays rather than mutating them.
  _snapshotRenderParams() {
    return {
      toneMidpoint:  RENDER_DEFAULTS.rotationToneMidpoint,
      autoNormalize: ROTATION_FIELD.autoNormalize,
      colorPositive: COLORS.rotationPositive,
      colorNegative: COLORS.rotationNegative,
      colorZero:     COLORS.rotationZero,
    };
  }

  _haveRenderParamsChanged() {
    const prev = this._lastRenderParamSnapshot;
    if (!prev) return true;
    return RENDER_DEFAULTS.rotationToneMidpoint !== prev.toneMidpoint  ||
           ROTATION_FIELD.autoNormalize         !== prev.autoNormalize ||
           COLORS.rotationPositive              !== prev.colorPositive ||
           COLORS.rotationNegative              !== prev.colorNegative ||
           COLORS.rotationZero                  !== prev.colorZero;
  }

  _haveComputeParamsChanged() {
    const prev = this._lastComputeParamSnapshot;
    if (!prev) return true;
    const curr = this._snapshotComputeParams();
    return curr.pairDistance  !== prev.pairDistance  ||
           curr.distanceDelta !== prev.distanceDelta ||
           curr.sampleStride  !== prev.sampleStride  ||
           curr.boundaryMode  !== prev.boundaryMode  ||
           curr.showMask      !== prev.showMask      ||
           curr.maskCenter    !== prev.maskCenter     ||
           curr.maskRadius    !== prev.maskRadius;
  }

  // Returns the mask bounding box in grid-cell space, clamped to the grid.
  // maskRadius is MOUSE_DEFAULTS.impulseRadius (grid cells).
  _resolveMaskBox() {
    const center = ROTATION_FIELD.maskCenter;
    if (!center) return { active: false };

    const r       = ROTATION_FIELD.maskRadius;
    const boxSize = Math.min(this._gridSize, Math.ceil(2 * r + 2));
    const halfBox = Math.floor(boxSize / 2);
    const originX = Math.max(0, Math.min(this._gridSize - boxSize, Math.round(center[0]) - halfBox));
    const originY = Math.max(0, Math.min(this._gridSize - boxSize, Math.round(center[1]) - halfBox));
    return { active: true, originX, originY, boxSize, centerX: center[0], centerY: center[1], radiusSq: r * r };
  }

  // Rebuilds the annulus span table on the GPU when pair parameters change.
  // Only the half plane dRow > 0 || (dRow == 0 && dCol > 0) is listed, so each unordered
  // pair is visited once; consecutive dCol of one dRow form a span [dColStart, dColEnd].
  _rebuildAnnulusSpans(pairDistance, distanceDelta) {
    const halfGrid    = this._gridSize / 2;
    const minRadius   = Math.max(0, (pairDistance - distanceDelta) * halfGrid);
    const maxRadius   = (pairDistance + distanceDelta) * halfGrid;
    const minRadiusSq = minRadius * minRadius;
    const maxRadiusSq = maxRadius * maxRadius;
    const isListed = (dCol, dRow) => {
      const distSq = dRow * dRow + dCol * dCol;
      return (dRow > 0 || dCol > 0) && distSq >= minRadiusSq && distSq <= maxRadiusSq;
    };

    const flat = [];
    // Loop bound clamped to halfGrid so each torus pair appears at most once, and floored
    // so odd grid sizes still enumerate integer offsets.
    // maxRadiusSq is intentionally NOT clamped — this allows offsets like (128,1)
    // at dist=128.004 to be included when the annulus extends slightly beyond halfGrid.
    const iMax = Math.min(Math.ceil(maxRadius), Math.floor(halfGrid));
    for (let dRow = 0; dRow <= iMax; dRow++) {
      let spanStart = null;
      for (let dCol = -iMax; dCol <= iMax; dCol++) {
        const listed = isListed(dCol, dRow);
        if (listed && spanStart === null) spanStart = dCol;
        if (!listed && spanStart !== null) {
          flat.push(dRow, spanStart, dCol - 1, 0);
          spanStart = null;
        }
      }
      if (spanStart !== null) flat.push(dRow, spanStart, iMax, 0);
    }

    if (flat.length > 0) {
      this._device.queue.writeBuffer(this._annulusSpanBuffer, 0, new Int32Array(flat));
    }
    this._annulusSpanCount = flat.length / SPAN_INTS;
    // Span buffer content changed — cached compute bind groups are stale.
    this._computeBindGroupCache.clear();
    this._annulusMinRadius = minRadius;
  }

  _writeComputeParams() {
    // Rebuild annulus span table when pair distance parameters change.
    const pairDistance  = ROTATION_FIELD.pairDistance;
    const distanceDelta = ROTATION_FIELD.distanceDelta;
    if (pairDistance !== this._lastPairDistance || distanceDelta !== this._lastDistanceDelta) {
      this._rebuildAnnulusSpans(pairDistance, distanceDelta);
      this._lastPairDistance  = pairDistance;
      this._lastDistanceDelta = distanceDelta;
    }

    // Field order must match struct Params in rotation_compute.wgsl exactly.
    // accumulationScale is compensated each frame:
    //   × sampleStride²      — stride S reduces contributing pairs by S²
    //   × gridSize / refGrid — larger grids have longer arms → omega ∝ 1/N
    const stride = ROTATION_FIELD.sampleStride;
    const compensatedScale = ROTATION_FIELD.accumulationScale
      * stride * stride
      * (this._gridSize / ROTATION_REFERENCE_GRID_SIZE);

    const mask = this._resolveMaskBox();

    new UniformWriter(64)
      .u32(this._gridSize)                          // gridSize
      .f32(compensatedScale)                        // accumulationScale
      .f32(ROTATION_FIELD.parallelThreshold)        // parallelThreshold
      .u32(this._annulusSpanCount)                  // spanCount
      .f32(this._annulusMinRadius)                  // minRadius
      .pad()                                        // pad
      .u32(stride)                                  // sampleStride
      .u32(mask.active ? mask.originX  : 0)         // maskOriginX
      .u32(mask.active ? mask.originY  : 0)         // maskOriginY
      .u32(mask.active ? mask.boxSize  : 0)         // maskBoxSize
      .f32(mask.active ? mask.centerX  : 0)         // maskCenterX
      .f32(mask.active ? mask.centerY  : 0)         // maskCenterY
      .f32(mask.active ? mask.radiusSq : 0)         // maskRadiusSq
      .u32(mask.active ? 1 : 0)                     // useMask
      .u32(PHYSICS_DEFAULTS.boundaryMode)           // boundaryMode
      .u32(ROTATION_FIELD.showMask)                // showMask
      .writeBuffer(this._device, this._computeParamsBuffer);
  }

  _writeReduceParams() {
    // Field order must match struct ReduceParams in rotation_reduce.wgsl exactly.
    new UniformWriter(16)
      .u32(this._gridSize)  // gridSize
      .pad()                // pad
      .pad()                // pad
      .pad()                // pad
      .writeBuffer(this._device, this._reduceParamsBuffer);
  }

  _writeEnergyParams() {
    // Field order must match struct EnergyParams in rotation_energy.wgsl exactly.
    new UniformWriter(16)
      .u32(this._gridSize * this._gridSize)  // totalCells
      .f32(0.02)                             // emaAlpha  (~35-frame half-life at 60 fps)
      .pad()                                 // pad
      .pad()                                 // pad
      .writeBuffer(this._device, this._energyParamsBuffer);
  }

  _writeRenderParams() {
    // Field order must match struct RenderParams in rotation_render.wgsl exactly.
    new UniformWriter(64)
      .u32(this._gridSize)                       // gridSize
      .f32(RENDER_DEFAULTS.rotationToneMidpoint) // rotationToneMidpoint
      .u32(ROTATION_FIELD.autoNormalize)         // autoNormalize
      .pad()                                     // pad
      .f32(COLORS.rotationPositive[0])           // colorPositive.r
      .f32(COLORS.rotationPositive[1])           // colorPositive.g
      .f32(COLORS.rotationPositive[2])           // colorPositive.b
      .pad()                                     // pad (vec3f → vec4f alignment)
      .f32(COLORS.rotationNegative[0])           // colorNegative.r
      .f32(COLORS.rotationNegative[1])           // colorNegative.g
      .f32(COLORS.rotationNegative[2])           // colorNegative.b
      .pad()                                     // pad (vec3f → vec4f alignment)
      .f32(COLORS.rotationZero[0])               // colorZero.r
      .f32(COLORS.rotationZero[1])               // colorZero.g
      .f32(COLORS.rotationZero[2])               // colorZero.b
      .pad()                                     // pad (vec3f → vec4f alignment)
      .writeBuffer(this._device, this._renderParamsBuffer);
  }
}

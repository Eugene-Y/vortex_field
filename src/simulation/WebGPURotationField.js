'use strict';

import { GRID_SIZE, ROTATION_FIELD, ROTATION_REFERENCE_GRID_SIZE, RENDER_DEFAULTS, COLORS, PHYSICS_DEFAULTS } from '../config/SimulationConfig.js';
import { UniformWriter } from '../gpu/UniformWriter.js';

const WORKGROUP_SIZE       = 64;
const ACCUMULATION_BUFFERS = 16;

/**
 * Field B — WebGPU implementation.
 *
 * Three passes per frame:
 *   1. Compute pass: N² threads, each iterates the precomputed annulus offset table,
 *      accumulates signed ω into one of ACCUMULATION_BUFFERS via fixed-point atomicAdd.
 *   2. Reduce pass: sum K buffers → flat f32 output buffer.
 *   3. Render pass: output buffer → canvas-rotation.
 *
 * External interface:
 *   recomputeFrom(velocityGpuTexture)  — takes current velocity GPUTexture directly
 *   dispose()
 */
export class WebGPURotationField {
  constructor(device, canvas, computeShaderSource, reduceShaderSource, renderShaderSource) {
    this._device   = device;
    this._canvas   = canvas;
    this._gridSize = GRID_SIZE;

    const totalCells = GRID_SIZE * GRID_SIZE;

    this._context = canvas.getContext('webgpu');
    const format  = navigator.gpu.getPreferredCanvasFormat();
    this._context.configure({ device, format, alphaMode: 'opaque' });

    // K atomic i32 accumulation buffers — cleared each frame.
    this._atomicBuffer = device.createBuffer({
      size:  ACCUMULATION_BUFFERS * totalCells * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // Flat f32 buffer: reduction output, also used directly by render shader.
    this._outputBuffer = device.createBuffer({
      size:  totalCells * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
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
      size:  48,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Precomputed annulus offset table: flat i32 pairs [dCol, dRow, ...].
    // Sized for the worst case: entire inscribed circle = π/4 × gridSize² offsets.
    // gridSize² × 8 bytes is a safe upper bound.
    this._annulusOffsetBuffer = device.createBuffer({
      size:  Math.max(8, totalCells * 8),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this._annulusOffsetCount = 0;
    this._annulusMinRadius   = 0;
    this._annulusMaxRadius   = 0;
    // Sentinel values that force a rebuild on the first frame.
    this._lastPairDistance  = -1;
    this._lastDistanceDelta = -1;

    this._computePipeline = this._buildComputePipeline(computeShaderSource);
    this._reducePipeline  = this._buildReducePipeline(reduceShaderSource);
    this._renderPipeline  = this._buildRenderPipeline(renderShaderSource, format);

    // Reduce bind group is stable (doesn't depend on the velocity texture).
    this._reduceBindGroup = this._buildReduceBindGroup();

    this._writeReduceParams();
    this._writeRenderParams();
  }

  /**
   * Runs compute+reduce+render passes using the supplied velocity GPUTexture.
   * Called once per frame from the main render loop.
   */
  recomputeFrom(velocityGpuTexture) {
    // Re-upload params each frame so slider changes take effect immediately.
    // _writeComputeParams also rebuilds the annulus table when pair params change.
    this._writeComputeParams();
    this._writeRenderParams();

    const totalCells  = this._gridSize * this._gridSize;
    const encoder     = this._device.createCommandEncoder();

    // Clear atomic buffers and output buffer before accumulation.
    encoder.clearBuffer(this._atomicBuffer);
    encoder.clearBuffer(this._outputBuffer);

    // Compute pass: accumulate rotation contributions.
    // Rebuild bind group each frame — velocity texture ping-pong changes which texture is current.
    const computeBindGroup = this._buildComputeBindGroup(velocityGpuTexture);
    const computePass = encoder.beginComputePass();
    computePass.setPipeline(this._computePipeline);
    computePass.setBindGroup(0, computeBindGroup);
    computePass.dispatchWorkgroups(Math.ceil(this._computeThreadCount() / WORKGROUP_SIZE));
    computePass.end();

    // Reduce pass: sum K buffers → output buffer.
    const reducePass = encoder.beginComputePass();
    reducePass.setPipeline(this._reducePipeline);
    reducePass.setBindGroup(0, this._reduceBindGroup);
    reducePass.dispatchWorkgroups(Math.ceil(totalCells / WORKGROUP_SIZE));
    reducePass.end();

    // Render pass: output buffer → canvas.
    const renderBindGroup = this._buildRenderBindGroup();
    const renderPass = encoder.beginRenderPass({
      colorAttachments: [{
        view:       this._context.getCurrentTexture().createView(),
        loadOp:     'clear',
        storeOp:    'store',
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
      }],
    });
    renderPass.setPipeline(this._renderPipeline);
    renderPass.setBindGroup(0, renderBindGroup);
    renderPass.draw(4);
    renderPass.end();

    this._device.queue.submit([encoder.finish()]);
  }

  dispose() {
    this._atomicBuffer.destroy();
    this._outputBuffer.destroy();
    this._computeParamsBuffer.destroy();
    this._reduceParamsBuffer.destroy();
    this._renderParamsBuffer.destroy();
    this._annulusOffsetBuffer.destroy();
  }

  _buildComputePipeline(shaderSource) {
    const module = this._device.createShaderModule({ code: shaderSource });
    return this._device.createComputePipeline({
      layout: 'auto',
      compute: {
        module,
        entryPoint: 'computeRotation',
        constants:  { ACCUMULATION_BUFFERS },
      },
    });
  }

  _buildReducePipeline(shaderSource) {
    const module = this._device.createShaderModule({ code: shaderSource });
    return this._device.createComputePipeline({
      layout: 'auto',
      compute: { module, entryPoint: 'reduceBuffers' },
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
    return this._device.createBindGroup({
      layout: this._computePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: velocityTexture.createView() },
        { binding: 1, resource: { buffer: this._atomicBuffer } },
        { binding: 2, resource: { buffer: this._computeParamsBuffer } },
        { binding: 3, resource: { buffer: this._annulusOffsetBuffer } },
      ],
    });
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

  _buildRenderBindGroup() {
    return this._device.createBindGroup({
      layout: this._renderPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this._outputBuffer } },
        { binding: 1, resource: { buffer: this._renderParamsBuffer } },
      ],
    });
  }

  // Returns the number of compute threads to dispatch this frame.
  // When mask is active, dispatches only the bounding-box cells (O(R²) vs O(N²)).
  _computeThreadCount() {
    const mask = this._resolveMaskBox();
    return mask.active ? mask.boxSize * mask.boxSize : this._gridSize * this._gridSize;
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

  // Rebuilds the flat i32 annulus offset table on the GPU when pair parameters change.
  // Stores the resulting count and radius bounds for use in the uniform write.
  _rebuildAnnulusOffsets(pairDistance, distanceDelta) {
    const halfGrid    = this._gridSize / 2;
    const minRadius   = Math.max(0,        (pairDistance - distanceDelta) * halfGrid);
    const maxRadius   = Math.min(halfGrid, (pairDistance + distanceDelta) * halfGrid);
    const minRadiusSq = minRadius * minRadius;
    const maxRadiusSq = maxRadius * maxRadius;

    const flat  = [];
    const iMax  = Math.ceil(maxRadius);
    for (let dRow = -iMax; dRow <= iMax; dRow++) {
      const dRowSq = dRow * dRow;
      if (dRowSq > maxRadiusSq) continue;
      for (let dCol = -iMax; dCol <= iMax; dCol++) {
        const distSq = dRowSq + dCol * dCol;
        if (distSq >= minRadiusSq && distSq <= maxRadiusSq) {
          flat.push(dCol, dRow);
        }
      }
    }

    if (flat.length > 0) {
      this._device.queue.writeBuffer(this._annulusOffsetBuffer, 0, new Int32Array(flat));
    }
    this._annulusOffsetCount = flat.length / 2;
    this._annulusMinRadius   = minRadius;
    this._annulusMaxRadius   = maxRadius;
  }

  _writeComputeParams() {
    // Rebuild annulus offset table when pair distance parameters change.
    const pairDistance  = ROTATION_FIELD.pairDistance;
    const distanceDelta = ROTATION_FIELD.distanceDelta;
    if (pairDistance !== this._lastPairDistance || distanceDelta !== this._lastDistanceDelta) {
      this._rebuildAnnulusOffsets(pairDistance, distanceDelta);
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
      .u32(this._annulusOffsetCount)                // offsetCount
      .f32(this._annulusMinRadius)                  // minRadius
      .f32(this._annulusMaxRadius)                  // maxRadius
      .u32(stride)                                  // sampleStride
      .u32(mask.active ? mask.originX  : 0)         // maskOriginX
      .u32(mask.active ? mask.originY  : 0)         // maskOriginY
      .u32(mask.active ? mask.boxSize  : 0)         // maskBoxSize
      .f32(mask.active ? mask.centerX  : 0)         // maskCenterX
      .f32(mask.active ? mask.centerY  : 0)         // maskCenterY
      .f32(mask.active ? mask.radiusSq : 0)         // maskRadiusSq
      .u32(mask.active ? 1 : 0)                     // useMask
      .u32(PHYSICS_DEFAULTS.boundaryMode)           // boundaryMode
      .pad()                                        // pad1
      .writeBuffer(this._device, this._computeParamsBuffer);
  }

  _writeReduceParams() {
    // Field order must match struct ReduceParams in rotation_reduce.wgsl exactly.
    new UniformWriter(16)
      .u32(this._gridSize)        // gridSize
      .u32(ACCUMULATION_BUFFERS)  // bufferCount
      .pad()                      // pad
      .pad()                      // pad
      .writeBuffer(this._device, this._reduceParamsBuffer);
  }

  _writeRenderParams() {
    // Field order must match struct RenderParams in rotation_render.wgsl exactly.
    new UniformWriter(48)
      .u32(this._gridSize)                       // gridSize
      .f32(RENDER_DEFAULTS.rotationToneMidpoint) // rotationToneMidpoint
      .pad()                                     // pad (vec2f alignment)
      .pad()                                     // pad
      .f32(COLORS.rotationPositive[0])           // colorPositive.r
      .f32(COLORS.rotationPositive[1])           // colorPositive.g
      .f32(COLORS.rotationPositive[2])           // colorPositive.b
      .pad()                                     // pad (vec3f → vec4f alignment)
      .f32(COLORS.rotationNegative[0])           // colorNegative.r
      .f32(COLORS.rotationNegative[1])           // colorNegative.g
      .f32(COLORS.rotationNegative[2])           // colorNegative.b
      .pad()                                     // pad (vec3f → vec4f alignment)
      .writeBuffer(this._device, this._renderParamsBuffer);
  }
}

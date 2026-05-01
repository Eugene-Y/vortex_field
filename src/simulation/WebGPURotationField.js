'use strict';

import { GRID_SIZE, ROTATION_FIELD, ROTATION_REFERENCE_GRID_SIZE, RENDER_DEFAULTS, COLORS, PHYSICS_DEFAULTS } from '../config/SimulationConfig.js';
import { VelocityBridge } from '../gpu/VelocityBridge.js';
import { UniformWriter } from '../gpu/UniformWriter.js';

const WORKGROUP_SIZE       = 64;
const ACCUMULATION_BUFFERS = 16;

/**
 * Field B — WebGPU implementation.
 *
 * Replaces the WebGL vertex-shader trick with a proper compute pipeline:
 *   1. Upload velocity from WebGL via VelocityBridge (readPixels → writeTexture)
 *   2. Compute pass: N² threads, each iterates j-neighbors in pairRange radius,
 *      accumulates signed ω into one of ACCUMULATION_BUFFERS via fixed-point atomicAdd.
 *   3. Reduce pass: sum K buffers → flat f32 output buffer.
 *   4. Render pass: output buffer → canvas-rotation.
 *
 * External interface matches old RotationField:
 *   recomputeFrom(velocityWebGLTexture, velocityFramebuffer)
 *   dispose()
 */
export class WebGPURotationField {
  constructor(device, gl, canvas, computeShaderSource, reduceShaderSource, renderShaderSource) {
    this._device  = device;
    this._gl      = gl;
    this._canvas  = canvas;
    this._gridSize = GRID_SIZE;

    const totalCells = GRID_SIZE * GRID_SIZE;

    this._context = canvas.getContext('webgpu');
    const format  = navigator.gpu.getPreferredCanvasFormat();
    this._context.configure({ device, format, alphaMode: 'opaque' });

    this._bridge = new VelocityBridge(gl, device, GRID_SIZE);

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

    this._computePipeline = this._buildComputePipeline(computeShaderSource);
    this._reducePipeline  = this._buildReducePipeline(reduceShaderSource);
    this._renderPipeline  = this._buildRenderPipeline(renderShaderSource, format);

    this._computeBindGroup = this._buildComputeBindGroup();
    this._reduceBindGroup  = this._buildReduceBindGroup();

    this._writeComputeParams();
    this._writeReduceParams();
    this._writeRenderParams();
  }

  /**
   * Uploads velocity from WebGL, runs compute+reduce+render passes.
   * Called once per frame from the main render loop.
   */
  recomputeFrom(velocityWebGLTexture, velocityFramebuffer) {
    this._bridge.upload(velocityWebGLTexture, velocityFramebuffer);

    // Re-upload params each frame so slider changes take effect immediately.
    this._writeComputeParams();
    this._writeRenderParams();

    const totalCells  = this._gridSize * this._gridSize;
    const encoder     = this._device.createCommandEncoder();

    // Clear atomic buffers and output buffer before accumulation.
    encoder.clearBuffer(this._atomicBuffer);
    encoder.clearBuffer(this._outputBuffer);

    // Compute pass: accumulate rotation contributions.
    const computePass = encoder.beginComputePass();
    computePass.setPipeline(this._computePipeline);
    computePass.setBindGroup(0, this._computeBindGroup);
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
    this._bridge.dispose();
    this._atomicBuffer.destroy();
    this._outputBuffer.destroy();
    this._computeParamsBuffer.destroy();
    this._reduceParamsBuffer.destroy();
    this._renderParamsBuffer.destroy();
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

  _buildComputeBindGroup() {
    return this._device.createBindGroup({
      layout: this._computePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this._bridge.gpuTexture.createView() },
        { binding: 1, resource: { buffer: this._atomicBuffer } },
        { binding: 2, resource: { buffer: this._computeParamsBuffer } },
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

  _writeComputeParams() {
    // Field order must match struct Params in rotation_compute.wgsl exactly.
    // accumulationScale is compensated each frame:
    //   × sampleStride²      — stride S reduces contributing pairs by S²
    //   × gridSize / refGrid — larger grids have longer arms → omega ∝ 1/N
    const stride = ROTATION_FIELD.sampleStride;
    const compensatedScale = ROTATION_FIELD.accumulationScale
      * stride * stride
      * (this._gridSize / ROTATION_REFERENCE_GRID_SIZE);

    const mask = this._resolveMaskBox();

    const writer = new UniformWriter(64);
    writer
      .u32(this._gridSize)                          // gridSize
      .f32(compensatedScale)                        // accumulationScale
      .f32(ROTATION_FIELD.parallelThreshold)        // parallelThreshold
      .f32(ROTATION_FIELD.pairRange)                // pairRange
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
      .pad()                                        // pad2
      .pad();                                       // pad3
    this._device.queue.writeBuffer(this._computeParamsBuffer, 0, writer.result());
  }

  _writeReduceParams() {
    // Field order must match struct ReduceParams in rotation_reduce.wgsl exactly.
    const writer = new UniformWriter(16);
    writer
      .u32(this._gridSize)        // gridSize
      .u32(ACCUMULATION_BUFFERS)  // bufferCount
      .pad()                      // pad
      .pad();                     // pad
    this._device.queue.writeBuffer(this._reduceParamsBuffer, 0, writer.result());
  }

  _writeRenderParams() {
    // Field order must match struct RenderParams in rotation_render.wgsl exactly.
    const writer = new UniformWriter(48);
    writer
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
      .pad();                                    // pad (vec3f → vec4f alignment)
    this._device.queue.writeBuffer(this._renderParamsBuffer, 0, writer.result());
  }
}

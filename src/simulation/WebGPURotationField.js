'use strict';

import { GRID_SIZE, ROTATION_FIELD, ROTATION_REFERENCE_GRID_SIZE, RENDER_DEFAULTS, COLORS } from '../config/SimulationConfig.js';
import { VelocityBridge } from '../gpu/VelocityBridge.js';

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
      size:  32,
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
    computePass.dispatchWorkgroups(Math.ceil(totalCells / WORKGROUP_SIZE));
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

  _writeComputeParams() {
    // struct Params: gridSize(u32), accumulationScale(f32), parallelThreshold(f32),
    //                pairRange(f32), sampleStride(u32), _pad×3 — 32 bytes
    //
    // accumulationScale is compensated each frame:
    //   × sampleStride²       — stride S reduces contributing pairs by S²
    //   × gridSize / refGrid  — larger grids have longer arms → omega ∝ 1/N
    const stride = ROTATION_FIELD.sampleStride;
    const compensatedScale = ROTATION_FIELD.accumulationScale
      * stride * stride
      * (this._gridSize / ROTATION_REFERENCE_GRID_SIZE);

    const data = new ArrayBuffer(32);
    const u32  = new Uint32Array(data);
    const f32  = new Float32Array(data);
    u32[0] = this._gridSize;
    f32[1] = compensatedScale;
    f32[2] = ROTATION_FIELD.parallelThreshold;
    f32[3] = ROTATION_FIELD.pairRange;
    u32[4] = stride;
    this._device.queue.writeBuffer(this._computeParamsBuffer, 0, data);
  }

  _writeReduceParams() {
    // struct ReduceParams: gridSize(u32), bufferCount(u32), _pad×2 — 16 bytes
    const data = new ArrayBuffer(16);
    const u32  = new Uint32Array(data);
    u32[0] = this._gridSize;
    u32[1] = ACCUMULATION_BUFFERS;
    this._device.queue.writeBuffer(this._reduceParamsBuffer, 0, data);
  }

  _writeRenderParams() {
    // struct RenderParams: gridSize(u32), rotationToneMidpoint(f32), _pad(vec2f),
    //                      colorPositive(vec3f), _pad(f32), colorNegative(vec3f), _pad(f32)
    // = 48 bytes
    const data = new ArrayBuffer(48);
    const u32  = new Uint32Array(data);
    const f32  = new Float32Array(data);
    u32[0] = this._gridSize;
    f32[1] = RENDER_DEFAULTS.rotationToneMidpoint;
    // pad: f32[2], f32[3]
    f32[4] = COLORS.rotationPositive[0];
    f32[5] = COLORS.rotationPositive[1];
    f32[6] = COLORS.rotationPositive[2];
    // pad: f32[7]
    f32[8]  = COLORS.rotationNegative[0];
    f32[9]  = COLORS.rotationNegative[1];
    f32[10] = COLORS.rotationNegative[2];
    this._device.queue.writeBuffer(this._renderParamsBuffer, 0, data);
  }
}

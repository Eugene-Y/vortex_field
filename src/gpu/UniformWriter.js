'use strict';

/**
 * Sequential writer for WebGPU uniform buffers.
 *
 * Writes u32 and f32 fields in declaration order, mirroring the WGSL struct layout.
 * Eliminates manual index arithmetic — the field sequence here IS the struct definition.
 *
 * Padding fields must be written explicitly via pad() so the slot count stays
 * synchronised with the WGSL struct. Wrong field count → wrong total offset → detectable
 * mismatch between the ArrayBuffer size and the GPUBuffer size at writeBuffer time.
 *
 * Usage:
 *   const writer = new UniformWriter(64);
 *   writer.u32(gridSize).f32(scale).pad().pad();
 *   device.queue.writeBuffer(buffer, 0, writer.result());
 */
export class UniformWriter {
  constructor(byteSize) {
    this._buffer = new ArrayBuffer(byteSize);
    this._u32    = new Uint32Array(this._buffer);
    this._f32    = new Float32Array(this._buffer);
    this._index  = 0;
  }

  u32(value) { this._u32[this._index++] = value; return this; }
  f32(value) { this._f32[this._index++] = value; return this; }

  // Advances the write head by one 4-byte slot without writing a value.
  // Call once per padding field in the WGSL struct.
  pad() { this._index++; return this; }

  result() { return this._buffer; }

  // Writes the accumulated buffer to a GPUBuffer via device.queue.writeBuffer.
  // Returns this for chaining, though typically called at the end of the chain.
  writeBuffer(device, gpuBuffer) {
    device.queue.writeBuffer(gpuBuffer, 0, this._buffer);
    return this;
  }
}

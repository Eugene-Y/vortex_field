'use strict';

/**
 * Reads the velocity field from a WebGL RG32F texture each frame and uploads
 * it to a WebGPU texture for use by the compute shader.
 *
 * The GPU→CPU→GPU round-trip via readPixels is a necessary cost of running
 * WebGL and WebGPU side-by-side without a shared memory extension.
 * At GRID_SIZE=512 this is ~2MB (RG32F = 8 bytes/pixel).
 */
export class VelocityBridge {
  constructor(gl, device, gridSize) {
    this._gl = gl;
    this._device = device;
    this._gridSize = gridSize;

    // Reusable CPU-side buffer — allocated once, reused every frame.
    this._pixelBuffer = new Float32Array(gridSize * gridSize * 2);

    this._gpuTexture = device.createTexture({
      size:   [gridSize, gridSize, 1],
      format: 'rg32float',
      usage:  GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
  }

  get gpuTexture() {
    return this._gpuTexture;
  }

  /**
   * Reads the current velocity texture from WebGL and pushes it to the GPU.
   * Must be called after the WebGL physics step and before the WebGPU compute pass.
   */
  upload(velocityWebGLTexture, velocityFramebuffer) {
    const gl = this._gl;

    gl.bindFramebuffer(gl.FRAMEBUFFER, velocityFramebuffer);
    gl.readPixels(
      0, 0,
      this._gridSize, this._gridSize,
      gl.RG,
      gl.FLOAT,
      this._pixelBuffer,
    );
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    this._device.queue.writeTexture(
      { texture: this._gpuTexture },
      this._pixelBuffer,
      { bytesPerRow: this._gridSize * 2 * 4 },  // 2 channels × 4 bytes
      [this._gridSize, this._gridSize, 1],
    );
  }

  dispose() {
    this._gpuTexture.destroy();
  }
}

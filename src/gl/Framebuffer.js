'use strict';

export const WRAPPING_CLAMP  = 'clamp';
export const WRAPPING_REPEAT = 'repeat';

/**
 * A pair of floating-point textures used for ping-pong rendering.
 * One is the read source, the other is the write target; they swap each step.
 */
export class PingPongFramebuffer {
  constructor(gl, width, height, internalFormat, format, type, wrapping = WRAPPING_CLAMP) {
    this._gl = gl;
    this._buffers = [
      createTextureAndFramebuffer(gl, width, height, internalFormat, format, type, wrapping),
      createTextureAndFramebuffer(gl, width, height, internalFormat, format, type, wrapping),
    ];
    this._readIndex = 0;
  }

  get readTexture() {
    return this._buffers[this._readIndex].texture;
  }

  get readFramebuffer() {
    return this._buffers[this._readIndex].framebuffer;
  }

  get writeFramebuffer() {
    return this._buffers[1 - this._readIndex].framebuffer;
  }

  swap() {
    this._readIndex = 1 - this._readIndex;
  }

  dispose() {
    const gl = this._gl;
    for (const { texture, framebuffer } of this._buffers) {
      gl.deleteTexture(texture);
      gl.deleteFramebuffer(framebuffer);
    }
  }
}

/**
 * A single floating-point texture + framebuffer pair (no ping-pong).
 * Used for fields that are fully rewritten each frame (e.g. Field B).
 */
export class SingleFramebuffer {
  constructor(gl, width, height, internalFormat, format, type, wrapping = WRAPPING_CLAMP) {
    this._gl = gl;
    const { texture, framebuffer } = createTextureAndFramebuffer(
      gl, width, height, internalFormat, format, type, wrapping
    );
    this.texture = texture;
    this.framebuffer = framebuffer;
  }

  dispose() {
    this._gl.deleteTexture(this.texture);
    this._gl.deleteFramebuffer(this.framebuffer);
  }
}

function createTextureAndFramebuffer(gl, width, height, internalFormat, format, type, wrapping) {
  const glWrapping = wrapping === WRAPPING_REPEAT ? gl.REPEAT : gl.CLAMP_TO_EDGE;

  const texture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, width, height, 0, format, type, null);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, glWrapping);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, glWrapping);

  const framebuffer = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);

  const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
  if (status !== gl.FRAMEBUFFER_COMPLETE) {
    throw new Error(`Framebuffer incomplete: status 0x${status.toString(16)}`);
  }

  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.bindTexture(gl.TEXTURE_2D, null);

  return { texture, framebuffer };
}

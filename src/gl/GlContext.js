'use strict';

/**
 * Creates and configures a WebGL2 context on the given canvas.
 * Throws if WebGL2 is unavailable.
 */
export function createGlContext(canvas) {
  const gl = canvas.getContext('webgl2', {
    antialias: false,
    alpha: false,
    preserveDrawingBuffer: false,
  });

  if (!gl) {
    throw new Error('WebGL2 is not available in this browser.');
  }

  requireExtension(gl, 'EXT_color_buffer_float');
  // Required for additive blending into float framebuffers (rotation accumulation).
  requireExtension(gl, 'EXT_float_blend');

  return gl;
}

function requireExtension(gl, name) {
  const extension = gl.getExtension(name);
  if (!extension) {
    throw new Error(`Required WebGL extension "${name}" is not available.`);
  }
  return extension;
}

'use strict';

import { SingleFramebuffer } from '../gl/Framebuffer.js';
import { PairRotationKernel } from './PairRotationKernel.js';
import { GRID_SIZE } from '../config/SimulationConfig.js';

/**
 * Field B: the instantaneous rotation center field.
 * Reset to zero every frame, then accumulated from all N² velocity pairs.
 */
export class RotationField {
  constructor(gl, rotationVertSource, rotationFragSource) {
    this._gl = gl;
    this._framebuffer = new SingleFramebuffer(
      gl, GRID_SIZE, GRID_SIZE, gl.R32F, gl.RED, gl.FLOAT
    );
    this._kernel = new PairRotationKernel(gl, rotationVertSource, rotationFragSource);
  }

  get rotationTexture() {
    return this._framebuffer.texture;
  }

  /**
   * Resets the field to zero, then accumulates contributions from all pairs
   * in the given velocity texture.
   */
  recomputeFrom(velocityTexture) {
    this._clearField();
    this._kernel.accumulateInto(this._framebuffer.framebuffer, velocityTexture);
  }

  _clearField() {
    const gl = this._gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this._framebuffer.framebuffer);
    gl.viewport(0, 0, GRID_SIZE, GRID_SIZE);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
  }

  dispose() {
    this._kernel.dispose();
    this._framebuffer.dispose();
  }
}

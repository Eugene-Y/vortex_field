'use strict';

import { ShaderProgram } from '../gl/ShaderProgram.js';
import { GRID_SIZE, ROTATION_FIELD } from '../config/SimulationConfig.js';

const TOTAL_CELLS = GRID_SIZE * GRID_SIZE;
const TOTAL_PAIRS = TOTAL_CELLS * TOTAL_CELLS; // N² pairs, including self-pairs (discarded in shader)

/**
 * GPU kernel that computes instantaneous rotation centers for all cell pairs
 * and accumulates signed rotation magnitudes into a target framebuffer.
 *
 * Uses point rendering with additive blending: each pair emits one point
 * at its center-of-rotation grid coordinate. The fragment adds its signed
 * magnitude to whatever is already in that pixel.
 */
export class PairRotationKernel {
  constructor(gl, vertexSource, fragmentSource) {
    this._gl = gl;
    this._program = new ShaderProgram(gl, vertexSource, fragmentSource);

    // Empty VAO — vertex data is computed entirely from gl_VertexID in the shader.
    this._emptyVao = gl.createVertexArray();
  }

  /**
   * Renders all pair contributions into the currently bound framebuffer.
   * The caller is responsible for clearing the framebuffer before this call
   * and for setting up additive blending.
   */
  accumulateInto(targetFramebuffer, velocityTexture) {
    const gl = this._gl;

    gl.bindFramebuffer(gl.FRAMEBUFFER, targetFramebuffer);
    gl.viewport(0, 0, GRID_SIZE, GRID_SIZE);

    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE); // additive blending for accumulation

    this._program.bind();
    this._program.setUniform1i('u_velocity', 0);
    this._program.setUniform1i('u_gridSize', GRID_SIZE);
    this._program.setUniform1f('u_parallelThreshold', ROTATION_FIELD.parallelThreshold);
    this._program.setUniform1f('u_accumulationScale', ROTATION_FIELD.accumulationScale);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, velocityTexture);

    gl.bindVertexArray(this._emptyVao);
    gl.drawArrays(gl.POINTS, 0, TOTAL_PAIRS);
    gl.bindVertexArray(null);

    gl.disable(gl.BLEND);
  }

  dispose() {
    this._gl.deleteVertexArray(this._emptyVao);
    this._program.dispose();
  }
}

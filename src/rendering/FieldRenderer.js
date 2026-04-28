'use strict';

import { ShaderProgram } from '../gl/ShaderProgram.js';
import { COLORS, RENDER_DEFAULTS } from '../config/SimulationConfig.js';

const RENDER_MODE_VELOCITY = 0;

/**
 * Renders a field texture to the currently active viewport using a full-screen quad.
 * The caller sets up the viewport and framebuffer before calling render methods.
 */
export class FieldRenderer {
  constructor(gl, vertexSource, fragmentSource) {
    this._gl = gl;
    this._program = new ShaderProgram(gl, vertexSource, fragmentSource);
  }

  renderVelocityField(texture, quadVao) {
    this._renderField(texture, quadVao, RENDER_MODE_VELOCITY);
  }

  _renderField(texture, quadVao, mode) {
    const gl = this._gl;

    this._program.bind();
    this._program.setUniform1i('u_field', 0);
    this._program.setUniform1i('u_mode', mode);
    this._program.setUniform3f('u_colorPositive', ...COLORS.rotationPositive);
    this._program.setUniform3f('u_colorNegative', ...COLORS.rotationNegative);
    this._program.setUniform1f('u_velocityToneMidpoint', RENDER_DEFAULTS.velocityToneMidpoint);
    this._program.setUniform1f('u_rotationToneMidpoint', RENDER_DEFAULTS.rotationToneMidpoint);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, texture);

    gl.bindVertexArray(quadVao);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.bindVertexArray(null);
  }

  dispose() {
    this._program.dispose();
  }
}

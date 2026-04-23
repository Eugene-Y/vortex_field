'use strict';

import { ShaderProgram } from '../gl/ShaderProgram.js';
import { PingPongFramebuffer, SingleFramebuffer, WRAPPING_REPEAT } from '../gl/Framebuffer.js';
import { GRID_SIZE, PHYSICS_DEFAULTS } from '../config/SimulationConfig.js';

const TEXEL_SIZE = 1.0 / GRID_SIZE;

/**
 * Physics step implementation: incompressible Navier-Stokes.
 *
 * Interface contract (all physics steps must implement):
 *   step(deltaTime, quadVao)            — advance simulation by deltaTime seconds
 *   injectImpulse(position, direction, radius, strength, quadVao)
 *   get velocityTexture                 — current velocity field (WebGLTexture)
 *   dispose()
 */
export class NavierStokesStep {
  constructor(gl, shaderSources) {
    this._gl = gl;
    // Read directly from PHYSICS_DEFAULTS each frame so UI sliders take effect immediately.

    // Periodic (REPEAT) boundaries: fluid wraps around edges, no artificial walls.
    this._velocity = new PingPongFramebuffer(
      gl, GRID_SIZE, GRID_SIZE, gl.RG32F, gl.RG, gl.FLOAT, WRAPPING_REPEAT
    );
    this._pressure = new PingPongFramebuffer(
      gl, GRID_SIZE, GRID_SIZE, gl.R32F, gl.RED, gl.FLOAT, WRAPPING_REPEAT
    );
    this._divergence = new SingleFramebuffer(
      gl, GRID_SIZE, GRID_SIZE, gl.R32F, gl.RED, gl.FLOAT, WRAPPING_REPEAT
    );

    this._advectProgram        = new ShaderProgram(gl, shaderSources.vert, shaderSources.advect);
    this._diffuseProgram       = new ShaderProgram(gl, shaderSources.vert, shaderSources.diffuse);
    this._divergenceProgram    = new ShaderProgram(gl, shaderSources.vert, shaderSources.divergence);
    this._pressureProgram      = new ShaderProgram(gl, shaderSources.vert, shaderSources.pressure);
    this._subtractGradProgram  = new ShaderProgram(gl, shaderSources.vert, shaderSources.subtractGradient);
    this._injectImpulseProgram = new ShaderProgram(gl, shaderSources.vert, shaderSources.injectImpulse);
    this._noiseProgram         = new ShaderProgram(gl, shaderSources.vert, shaderSources.noise);
  }

  get velocityTexture() {
    return this._velocity.readTexture;
  }

  step(deltaTime, quadVao) {
    this._advect(deltaTime, quadVao);
    this._diffuse(deltaTime, quadVao);
    this._solvePressureAndProjectVelocity(quadVao);
  }

  injectImpulse(position, direction, radius, strength, quadVao) {
    const gl = this._gl;

    gl.bindFramebuffer(gl.FRAMEBUFFER, this._velocity.writeFramebuffer);
    gl.viewport(0, 0, GRID_SIZE, GRID_SIZE);

    this._injectImpulseProgram.bind();
    this._injectImpulseProgram.setUniform1i('u_velocity', 0);
    this._injectImpulseProgram.setUniform2f('u_impulsePosition', position[0], position[1]);
    this._injectImpulseProgram.setUniform2f('u_impulseDirection', direction[0], direction[1]);
    this._injectImpulseProgram.setUniform1f('u_impulseRadius', radius / GRID_SIZE);
    this._injectImpulseProgram.setUniform1f('u_impulseStrength', strength);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this._velocity.readTexture);

    gl.bindVertexArray(quadVao);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.bindVertexArray(null);

    this._velocity.swap();
  }

  // Adds per-cell random velocity. seed should differ each call for distinct noise.
  injectNoise(strength, seed, quadVao) {
    const gl = this._gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this._velocity.writeFramebuffer);
    gl.viewport(0, 0, GRID_SIZE, GRID_SIZE);

    this._noiseProgram.bind();
    this._noiseProgram.setUniform1i('u_velocity', 0);
    this._noiseProgram.setUniform1f('u_strength', strength);
    this._noiseProgram.setUniform1f('u_seed', seed);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this._velocity.readTexture);

    drawFullScreenQuad(gl, quadVao);
    this._velocity.swap();
  }

  clearVelocity() {
    const gl = this._gl;
    // Both ping-pong buffers must be cleared.
    for (let i = 0; i < 2; i++) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, this._velocity.writeFramebuffer);
      gl.clearBufferfv(gl.COLOR, 0, new Float32Array([0, 0, 0, 0]));
      this._velocity.swap();
    }
  }

  dispose() {
    this._velocity.dispose();
    this._pressure.dispose();
    this._divergence.dispose();
    this._advectProgram.dispose();
    this._diffuseProgram.dispose();
    this._divergenceProgram.dispose();
    this._pressureProgram.dispose();
    this._subtractGradProgram.dispose();
    this._injectImpulseProgram.dispose();
    this._noiseProgram.dispose();
  }

  _advect(deltaTime, quadVao) {
    const gl = this._gl;

    gl.bindFramebuffer(gl.FRAMEBUFFER, this._velocity.writeFramebuffer);
    gl.viewport(0, 0, GRID_SIZE, GRID_SIZE);

    this._advectProgram.bind();
    this._advectProgram.setUniform1i('u_velocity', 0);
    this._advectProgram.setUniform1f('u_deltaTime', deltaTime);
    this._advectProgram.setUniform1f('u_gridSize', GRID_SIZE);
    this._advectProgram.setUniform1f('u_damping', PHYSICS_DEFAULTS.damping);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this._velocity.readTexture);

    drawFullScreenQuad(gl, quadVao);
    this._velocity.swap();
  }

  _diffuse(deltaTime, quadVao) {
    const gl = this._gl;
    const alpha = (TEXEL_SIZE * TEXEL_SIZE) / (PHYSICS_DEFAULTS.viscosity * deltaTime);
    const beta = 1.0 / (4.0 + alpha);
    const prevTexture = this._velocity.readTexture;

    for (let iteration = 0; iteration < PHYSICS_DEFAULTS.diffusionIterations; iteration++) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, this._velocity.writeFramebuffer);
      gl.viewport(0, 0, GRID_SIZE, GRID_SIZE);

      this._diffuseProgram.bind();
      this._diffuseProgram.setUniform1i('u_velocity', 0);
      this._diffuseProgram.setUniform1i('u_velocityPrev', 1);
      this._diffuseProgram.setUniform1f('u_alpha', alpha);
      this._diffuseProgram.setUniform1f('u_beta', beta);
      this._diffuseProgram.setUniform1f('u_texelSize', TEXEL_SIZE);

      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this._velocity.readTexture);
      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, prevTexture);

      drawFullScreenQuad(gl, quadVao);
      this._velocity.swap();
    }
  }

  _solvePressureAndProjectVelocity(quadVao) {
    this._computeDivergence(quadVao);
    this._solvePressure(quadVao);
    this._subtractPressureGradient(quadVao);
  }

  _computeDivergence(quadVao) {
    const gl = this._gl;

    gl.bindFramebuffer(gl.FRAMEBUFFER, this._divergence.framebuffer);
    gl.viewport(0, 0, GRID_SIZE, GRID_SIZE);

    this._divergenceProgram.bind();
    this._divergenceProgram.setUniform1i('u_velocity', 0);
    this._divergenceProgram.setUniform1f('u_texelSize', TEXEL_SIZE);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this._velocity.readTexture);

    drawFullScreenQuad(gl, quadVao);
  }

  _solvePressure(quadVao) {
    const gl = this._gl;

    for (let iteration = 0; iteration < PHYSICS_DEFAULTS.pressureIterations; iteration++) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, this._pressure.writeFramebuffer);
      gl.viewport(0, 0, GRID_SIZE, GRID_SIZE);

      this._pressureProgram.bind();
      this._pressureProgram.setUniform1i('u_pressure', 0);
      this._pressureProgram.setUniform1i('u_divergence', 1);
      this._pressureProgram.setUniform1f('u_texelSize', TEXEL_SIZE);

      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this._pressure.readTexture);
      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, this._divergence.texture);

      drawFullScreenQuad(gl, quadVao);
      this._pressure.swap();
    }
  }

  _subtractPressureGradient(quadVao) {
    const gl = this._gl;

    gl.bindFramebuffer(gl.FRAMEBUFFER, this._velocity.writeFramebuffer);
    gl.viewport(0, 0, GRID_SIZE, GRID_SIZE);

    this._subtractGradProgram.bind();
    this._subtractGradProgram.setUniform1i('u_velocity', 0);
    this._subtractGradProgram.setUniform1i('u_pressure', 1);
    this._subtractGradProgram.setUniform1f('u_texelSize', TEXEL_SIZE);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this._velocity.readTexture);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this._pressure.readTexture);

    drawFullScreenQuad(gl, quadVao);
    this._velocity.swap();
  }
}

function drawFullScreenQuad(gl, quadVao) {
  gl.bindVertexArray(quadVao);
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  gl.bindVertexArray(null);
}

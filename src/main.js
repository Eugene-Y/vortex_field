'use strict';

import { createGlContext }  from './gl/GlContext.js';
import { FluidField }       from './simulation/FluidField.js';
import { RotationField }    from './simulation/RotationField.js';
import { FieldRenderer }    from './rendering/FieldRenderer.js';
import { MouseInjector }    from './interaction/MouseInjector.js';
import { GRID_SIZE, DISPLAY_SCALE } from './config/SimulationConfig.js';

const CANVAS_FIELD_SIZE = GRID_SIZE * DISPLAY_SCALE;

async function loadShaderSource(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`Failed to load shader: ${path}`);
  return response.text();
}

async function loadAllShaders() {
  const [
    commonVert,
    advectFrag,
    diffuseFrag,
    divergenceFrag,
    pressureFrag,
    subtractGradientFrag,
    injectImpulseFrag,
    rotationAccumulateVert,
    rotationAccumulateFrag,
    renderFrag,
  ] = await Promise.all([
    loadShaderSource('src/shaders/common.vert'),
    loadShaderSource('src/shaders/advect.frag'),
    loadShaderSource('src/shaders/diffuse.frag'),
    loadShaderSource('src/shaders/divergence.frag'),
    loadShaderSource('src/shaders/pressure.frag'),
    loadShaderSource('src/shaders/subtract_gradient.frag'),
    loadShaderSource('src/shaders/inject_impulse.frag'),
    loadShaderSource('src/shaders/rotation_accumulate.vert'),
    loadShaderSource('src/shaders/rotation_accumulate.frag'),
    loadShaderSource('src/shaders/render.frag'),
  ]);

  return {
    physics: {
      vert:             commonVert,
      advect:           advectFrag,
      diffuse:          diffuseFrag,
      divergence:       divergenceFrag,
      pressure:         pressureFrag,
      subtractGradient: subtractGradientFrag,
      injectImpulse:    injectImpulseFrag,
    },
    rotation: {
      vert: rotationAccumulateVert,
      frag: rotationAccumulateFrag,
    },
    render: {
      vert: commonVert,
      frag: renderFrag,
    },
  };
}

function configureCanvas(canvas) {
  // Two fields side by side in one canvas — single GL context, no cross-context texture sharing.
  canvas.width  = CANVAS_FIELD_SIZE * 2;
  canvas.height = CANVAS_FIELD_SIZE;
  canvas.style.width  = `${CANVAS_FIELD_SIZE * 2}px`;
  canvas.style.height = `${CANVAS_FIELD_SIZE}px`;
}

function createFullScreenQuadVao(gl) {
  const positions = new Float32Array([-1, -1,  1, -1,  -1, 1,  1, 1]);
  const vao = gl.createVertexArray();
  gl.bindVertexArray(vao);
  const buffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
  gl.bindVertexArray(null);
  return vao;
}

async function main() {
  const canvas = document.getElementById('canvas-main');
  configureCanvas(canvas);

  const gl = createGlContext(canvas);
  const shaders = await loadAllShaders();

  const fluidField    = new FluidField(gl, shaders.physics);
  const rotationField = new RotationField(gl, shaders.rotation.vert, shaders.rotation.frag);
  const renderer      = new FieldRenderer(gl, shaders.render.vert, shaders.render.frag);
  const quadVao       = createFullScreenQuadVao(gl);
  const mouseInjector = new MouseInjector(canvas, fluidField, CANVAS_FIELD_SIZE);

  let previousTime = performance.now();

  function renderFrame(currentTime) {
    const deltaTime = Math.min((currentTime - previousTime) / 1000, 0.05);
    previousTime = currentTime;

    mouseInjector.applyPendingInjection(quadVao);

    fluidField.step(deltaTime, quadVao);
    rotationField.recomputeFrom(fluidField.velocityTexture);

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    // Left half: Field A — velocity
    gl.viewport(0, 0, CANVAS_FIELD_SIZE, CANVAS_FIELD_SIZE);
    renderer.renderVelocityField(fluidField.velocityTexture, quadVao);

    // Right half: Field B — rotation centers
    gl.viewport(CANVAS_FIELD_SIZE, 0, CANVAS_FIELD_SIZE, CANVAS_FIELD_SIZE);
    renderer.renderRotationField(rotationField.rotationTexture, quadVao);

    requestAnimationFrame(renderFrame);
  }

  requestAnimationFrame(renderFrame);
}

main().catch(error => {
  console.error('Vortex field initialization failed:', error);
});

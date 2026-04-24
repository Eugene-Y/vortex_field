'use strict';

import { createGlContext }  from './gl/GlContext.js';
import { FluidField }       from './simulation/FluidField.js';
import { RotationField }    from './simulation/RotationField.js';
import { FieldRenderer }    from './rendering/FieldRenderer.js';
import { MouseInjector }    from './interaction/MouseInjector.js';
import { ControlPanel }     from './ui/ControlPanel.js';
import { PatternInjector }  from './interaction/PatternInjector.js';
import { GRID_SIZE, DISPLAY_SCALE, DISPLAY_GAP, PHYSICS_DEFAULTS, buildShareUrl } from './config/SimulationConfig.js';

const CANVAS_FIELD_SIZE = computeFieldPixelSize();

function computeFieldPixelSize() {
  const reservedVertical   = 180; // labels + sliders + margins
  const reservedHorizontal = 48;
  const maxFromWidth  = Math.floor((window.innerWidth  - reservedHorizontal - DISPLAY_GAP) / 2 / GRID_SIZE);
  const maxFromHeight = Math.floor((window.innerHeight - reservedVertical) / GRID_SIZE);
  const scale = Math.max(1, Math.min(DISPLAY_SCALE, maxFromWidth, maxFromHeight));
  return GRID_SIZE * scale;
}

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
    noiseFrag,
    vorticityCurlFrag,
    vorticityConfinementFrag,
    injectDiskFrag,
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
    loadShaderSource('src/shaders/noise.frag'),
    loadShaderSource('src/shaders/vorticity_curl.frag'),
    loadShaderSource('src/shaders/vorticity_confinement.frag'),
    loadShaderSource('src/shaders/inject_disk.frag'),
    loadShaderSource('src/shaders/rotation_accumulate.vert'),
    loadShaderSource('src/shaders/rotation_accumulate.frag'),
    loadShaderSource('src/shaders/render.frag'),
  ]);

  return {
    physics: {
      vert:                   commonVert,
      advect:                 advectFrag,
      diffuse:                diffuseFrag,
      divergence:             divergenceFrag,
      pressure:               pressureFrag,
      subtractGradient:       subtractGradientFrag,
      injectImpulse:          injectImpulseFrag,
      noise:                  noiseFrag,
      vorticityCurl:          vorticityCurlFrag,
      vorticityConfinement:   vorticityConfinementFrag,
      injectDisk:             injectDiskFrag,
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
  // Two fields side by side with a gap — single GL context, no cross-context texture sharing.
  canvas.width  = CANVAS_FIELD_SIZE * 2 + DISPLAY_GAP;
  canvas.height = CANVAS_FIELD_SIZE;
  canvas.style.width  = `${CANVAS_FIELD_SIZE * 2 + DISPLAY_GAP}px`;
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
  const mouseInjector    = new MouseInjector(canvas, fluidField, CANVAS_FIELD_SIZE);
  const velocityControls = document.getElementById('controls-velocity');
  const rotationControls = document.getElementById('controls-rotation');

  const controlPanel = new ControlPanel(
    velocityControls,
    rotationControls,
    velocityControls,
    CANVAS_FIELD_SIZE,
    DISPLAY_GAP,
  );

  const patternInjector = new PatternInjector(
    canvas, fluidField, CANVAS_FIELD_SIZE, DISPLAY_GAP,
    rotationControls,
  );

  controlPanel.addRotationSliders(rotationControls);

  patternInjector.injectAt([0.5, 0.5], quadVao);

  let previousTime = performance.now();
  let animationFrameId = null;
  let paused = false;

  function renderFields() {
    rotationField.recomputeFrom(fluidField.velocityTexture);

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);

    gl.viewport(0, 0, CANVAS_FIELD_SIZE, CANVAS_FIELD_SIZE);
    renderer.renderVelocityField(fluidField.velocityTexture, quadVao);

    gl.viewport(CANVAS_FIELD_SIZE + DISPLAY_GAP, 0, CANVAS_FIELD_SIZE, CANVAS_FIELD_SIZE);
    renderer.renderRotationField(rotationField.rotationTexture, quadVao);
  }

  function renderFrame(currentTime) {
    const baseDeltaTime = Math.min((currentTime - previousTime) / 1000, 0.05);
    const deltaTime = baseDeltaTime * PHYSICS_DEFAULTS.simulationSpeed;
    previousTime = currentTime;

    mouseInjector.applyPendingInjection(quadVao);
    patternInjector.applyPendingPattern(quadVao);

    fluidField.step(deltaTime, quadVao);
    renderFields();

    animationFrameId = requestAnimationFrame(renderFrame);
  }

  function stopLoop() {
    cancelAnimationFrame(animationFrameId);
    animationFrameId = null;
  }

  function startLoop() {
    previousTime = performance.now();
    animationFrameId = requestAnimationFrame(renderFrame);
  }

  function togglePause() {
    paused = !paused;
    const button = document.getElementById('pause-button');
    if (paused) {
      stopLoop();
      button.textContent = 'Play';
    } else {
      button.textContent = 'Pause';
      startLoop();
    }
  }

  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      stopLoop();
    } else if (!paused) {
      startLoop();
    }
  });

  document.getElementById('pause-button').addEventListener('click', togglePause);
  document.getElementById('clear-button').addEventListener('click', () => fluidField.reset());
  document.getElementById('reset-button').addEventListener('click', () => {
    window.location.href = window.location.pathname;
  });

  document.getElementById('copy-url-button').addEventListener('click', () => {
    navigator.clipboard.writeText(buildShareUrl());
  });

  document.addEventListener('input', () => {
    history.replaceState(null, '', buildShareUrl());
    if (paused) renderFields();
  });

  const gridSizeInput = document.getElementById('grid-size-input');
  gridSizeInput.value = GRID_SIZE;
  gridSizeInput.addEventListener('change', () => {
    const value = Math.max(32, Math.min(512, parseInt(gridSizeInput.value, 10) || GRID_SIZE));
    const params = new URLSearchParams(window.location.search);
    params.set('gridSize', value);
    window.location.search = params.toString();
  });

  animationFrameId = requestAnimationFrame(renderFrame);
}

main().catch(error => {
  console.error('Vortex field initialization failed:', error);
});

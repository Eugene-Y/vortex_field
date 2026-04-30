'use strict';

import { createGlContext }         from './gl/GlContext.js';
import { createWebGPUDevice }      from './gpu/WebGPUDevice.js';
import { FluidField }              from './simulation/FluidField.js';
import { WebGPURotationField }     from './simulation/WebGPURotationField.js';
import { FieldRenderer }           from './rendering/FieldRenderer.js';
import { MouseInjector }           from './interaction/MouseInjector.js';
import { ControlPanel }            from './ui/ControlPanel.js';
import { PatternInjector }         from './interaction/PatternInjector.js';
import { FocusMaskInteractor }     from './interaction/FocusMaskInteractor.js';
import { GRID_SIZE, DISPLAY_SCALE, DISPLAY_GAP, PHYSICS_DEFAULTS, buildShareUrl } from './config/SimulationConfig.js';

const CANVAS_FIELD_SIZE = computeFieldPixelSize();

function computeFieldPixelSize() {
  const reservedVertical   = 180;
  const reservedHorizontal = 48;
  const maxFromWidth  = Math.floor((window.innerWidth  - reservedHorizontal - DISPLAY_GAP) / 2 / GRID_SIZE);
  const maxFromHeight = Math.floor((window.innerHeight - reservedVertical) / GRID_SIZE);
  const scale = Math.max(1, Math.min(DISPLAY_SCALE, maxFromWidth, maxFromHeight));
  return GRID_SIZE * scale;
}

async function loadShaderSource(path) {
  const response = await fetch(path, { cache: 'reload' });
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
    renderFrag,
    rotationComputeWgsl,
    rotationReduceWgsl,
    rotationRenderWgsl,
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
    loadShaderSource('src/shaders/render.frag'),
    loadShaderSource('src/shaders/rotation_compute.wgsl'),
    loadShaderSource('src/shaders/rotation_reduce.wgsl'),
    loadShaderSource('src/shaders/rotation_render.wgsl'),
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
    render: {
      vert: commonVert,
      frag: renderFrag,
    },
    rotation: {
      compute: rotationComputeWgsl,
      reduce:  rotationReduceWgsl,
      render:  rotationRenderWgsl,
    },
  };
}

function configureCanvas(canvas) {
  canvas.width  = CANVAS_FIELD_SIZE;
  canvas.height = CANVAS_FIELD_SIZE;
  canvas.style.width  = `${CANVAS_FIELD_SIZE}px`;
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

function showWebGPUError(message) {
  const fields = document.getElementById('fields');
  const errorBox = document.createElement('div');
  errorBox.style.cssText = 'color:#f55;font-size:13px;padding:20px;max-width:500px;text-align:center;line-height:1.6';
  errorBox.textContent = `WebGPU unavailable: ${message}`;
  fields.replaceWith(errorBox);
}

async function main() {
  const canvasMain     = document.getElementById('canvas-main');
  const canvasRotation = document.getElementById('canvas-rotation');

  configureCanvas(canvasMain);
  configureCanvas(canvasRotation);

  // Set the gap between the two field wrappers to match DISPLAY_GAP.
  document.getElementById('fields').style.gap = `${DISPLAY_GAP}px`;

  let gpuDevice;
  try {
    gpuDevice = await createWebGPUDevice();
  } catch (error) {
    showWebGPUError(error.message);
    return;
  }

  const gl      = createGlContext(canvasMain);
  const shaders = await loadAllShaders();

  const fluidField    = new FluidField(gl, shaders.physics);
  const rotationField = new WebGPURotationField(
    gpuDevice, gl, canvasRotation,
    shaders.rotation.compute,
    shaders.rotation.reduce,
    shaders.rotation.render,
  );
  const renderer   = new FieldRenderer(gl, shaders.render.vert, shaders.render.frag);
  const quadVao    = createFullScreenQuadVao(gl);

  const mouseInjector    = new MouseInjector(canvasMain, fluidField, CANVAS_FIELD_SIZE);
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
    canvasMain, fluidField, CANVAS_FIELD_SIZE, DISPLAY_GAP,
    rotationControls, canvasRotation,
  );
  const focusMask = new FocusMaskInteractor(canvasRotation, canvasMain);

  controlPanel.addRotationSliders(rotationControls);
  patternInjector.queueInitialInjection([0.5, 0.5]);

  let previousTime    = performance.now();
  let animationFrameId = null;
  let paused          = false;

  function renderFields() {
    rotationField.recomputeFrom(fluidField.velocityTexture, fluidField.velocityFramebuffer);

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, CANVAS_FIELD_SIZE, CANVAS_FIELD_SIZE);
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    renderer.renderVelocityField(fluidField.velocityTexture, quadVao);

    focusMask.updateOverlay();
  }

  function renderFrame(currentTime) {
    const baseDeltaTime = Math.min((currentTime - previousTime) / 1000, 0.033);
    const deltaTime     = baseDeltaTime * PHYSICS_DEFAULTS.simulationSpeed;
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
    previousTime     = performance.now();
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
    const value = Math.max(32, Math.min(1024, parseInt(gridSizeInput.value, 10) || GRID_SIZE));
    const params = new URLSearchParams(window.location.search);
    params.set('gridSize', value);
    window.location.search = params.toString();
  });

  animationFrameId = requestAnimationFrame(renderFrame);
}

main().catch(error => {
  console.error('Vortex field initialization failed:', error);
});

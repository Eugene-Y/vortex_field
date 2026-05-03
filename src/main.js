'use strict';

import { createWebGPUDevice }      from './gpu/WebGPUDevice.js';
import { FluidField }              from './simulation/FluidField.js';
import { WebGPURotationField }     from './simulation/WebGPURotationField.js';
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
    velocityAdvect,
    velocityDivergence,
    velocityPressure,
    velocitySubtractGradient,
    velocityVorticityCurl,
    velocityVorticityConfinement,
    velocityInjectImpulse,
    velocityInjectDisk,
    velocityNoise,
    velocityRender,
    rotationCompute,
    rotationReduce,
    rotationRender,
  ] = await Promise.all([
    loadShaderSource('src/shaders/velocity_advect.wgsl'),
    loadShaderSource('src/shaders/velocity_divergence.wgsl'),
    loadShaderSource('src/shaders/velocity_pressure.wgsl'),
    loadShaderSource('src/shaders/velocity_subtract_gradient.wgsl'),
    loadShaderSource('src/shaders/velocity_vorticity_curl.wgsl'),
    loadShaderSource('src/shaders/velocity_vorticity_confinement.wgsl'),
    loadShaderSource('src/shaders/velocity_inject_impulse.wgsl'),
    loadShaderSource('src/shaders/velocity_inject_disk.wgsl'),
    loadShaderSource('src/shaders/velocity_noise.wgsl'),
    loadShaderSource('src/shaders/velocity_render.wgsl'),
    loadShaderSource('src/shaders/rotation_compute.wgsl'),
    loadShaderSource('src/shaders/rotation_reduce.wgsl'),
    loadShaderSource('src/shaders/rotation_render.wgsl'),
  ]);

  return {
    velocity: {
      advect:               velocityAdvect,
      divergence:           velocityDivergence,
      pressure:             velocityPressure,
      subtractGradient:     velocitySubtractGradient,
      vorticityCurl:        velocityVorticityCurl,
      vorticityConfinement: velocityVorticityConfinement,
      injectImpulse:        velocityInjectImpulse,
      injectDisk:           velocityInjectDisk,
      noise:                velocityNoise,
      render:               velocityRender,
    },
    rotation: {
      compute: rotationCompute,
      reduce:  rotationReduce,
      render:  rotationRender,
    },
  };
}

function configureCanvas(canvas) {
  canvas.width  = CANVAS_FIELD_SIZE;
  canvas.height = CANVAS_FIELD_SIZE;
  canvas.style.width  = `${CANVAS_FIELD_SIZE}px`;
  canvas.style.height = `${CANVAS_FIELD_SIZE}px`;
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

  const shaders = await loadAllShaders();

  const fluidField    = new FluidField(gpuDevice, canvasMain, shaders.velocity);
  const rotationField = new WebGPURotationField(
    gpuDevice, canvasRotation,
    shaders.rotation.compute,
    shaders.rotation.reduce,
    shaders.rotation.render,
  );

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

  // addRotationSliders must run first — it creates the pattern-size row that
  // PatternInjector appends the inject dropdown to.
  const patternSizeRow = controlPanel.addRotationSliders(rotationControls);
  const patternInjector = new PatternInjector(
    canvasMain, fluidField, CANVAS_FIELD_SIZE, DISPLAY_GAP,
    patternSizeRow, canvasRotation,
  );
  const focusMask = new FocusMaskInteractor(canvasRotation, canvasMain);
  patternInjector.queueInitialInjection([0.5, 0.5]);

  let animationFrameId = null;
  let paused           = false;

  function renderFields() {
    fluidField.render();
    rotationField.recomputeFrom(fluidField.velocityTexture, fluidField.stepGeneration);
    focusMask.updateOverlay();
  }

  function renderFrame() {
    // Fixed physics step decoupled from wall-clock time: GPU stalls from heavy
    // Field B compute cannot inflate deltaTime and inject physics spikes into Field A.
    const deltaTime = (1 / 60) * PHYSICS_DEFAULTS.simulationSpeed;

    mouseInjector.applyPendingInjection();
    patternInjector.applyPendingPattern();

    fluidField.step(deltaTime);
    renderFields();

    animationFrameId = requestAnimationFrame(renderFrame);
  }

  function stopLoop() {
    cancelAnimationFrame(animationFrameId);
    animationFrameId = null;
  }

  function startLoop() {
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

'use strict';

import { createWebGPUDevice }      from './gpu/WebGPUDevice.js';
import { FluidField }              from './simulation/FluidField.js';
import { WebGPURotationField }     from './simulation/WebGPURotationField.js';
import { MouseInjector }           from './interaction/MouseInjector.js';
import { ControlPanel }            from './ui/ControlPanel.js';
import { ControlHintDisplay }      from './ui/ControlHintDisplay.js';
import { PatternInjector }         from './interaction/PatternInjector.js';
import { FocusMaskInteractor }     from './interaction/FocusMaskInteractor.js';
import { FieldSnapshotSaver }      from './interaction/FieldSnapshotSaver.js';
import { RandomSceneShortcut }     from './interaction/RandomSceneShortcut.js';
import { blockSelectTypeAhead }    from './ui/SelectTypeAheadBlocker.js';
import { randomizeSceneSettings }  from './config/RandomSceneSettings.js';
import { GRID_SIZE, DISPLAY_SCALE, DISPLAY_GAP, PHYSICS_DEFAULTS, GRAVITY_EXPERIMENT, START_PULSES, OPENED_FROM_CANONICAL_LINK, buildShareUrl, buildDefaultSettingsUrl } from './config/SimulationConfig.js';

const RESERVED_VERTICAL_PX   = 180;
const RESERVED_HORIZONTAL_PX = 48;

const START_PATTERN_CENTER = [0.5, 0.5];

const FIELD_LAYOUT      = computeFieldLayout();
const CANVAS_FIELD_SIZE = FIELD_LAYOUT.fieldPixelSize;

// Side by side when both fields fit the window width at one pixel per cell; otherwise each
// field with its controls is stacked under the other and the page scrolls vertically.
function computeFieldLayout() {
  const availableWidth  = window.innerWidth  - RESERVED_HORIZONTAL_PX;
  const availableHeight = window.innerHeight - RESERVED_VERTICAL_PX;
  const scaleFromHeight     = Math.floor(availableHeight / GRID_SIZE);
  const sideBySideFromWidth = Math.floor((availableWidth - DISPLAY_GAP) / 2 / GRID_SIZE);
  const stacked        = sideBySideFromWidth < 1;
  const scaleFromWidth = stacked ? Math.floor(availableWidth / GRID_SIZE) : sideBySideFromWidth;
  const scale = Math.max(1, Math.min(DISPLAY_SCALE, scaleFromWidth, scaleFromHeight));
  return { fieldPixelSize: GRID_SIZE * scale, stacked };
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
    noiseLattice,
    velocityRender,
    densityAdvect,
    gravityPoisson,
    gravityForce,
    densityInjectImpulse,
    densityInjectDisk,
    densityInjectNoise,
    rotationCompute,
    rotationReduce,
    rotationEnergy,
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
    loadShaderSource('src/shaders/noise_lattice.wgsl'),
    loadShaderSource('src/shaders/velocity_render.wgsl'),
    loadShaderSource('src/shaders/density_advect.wgsl'),
    loadShaderSource('src/shaders/gravity_poisson.wgsl'),
    loadShaderSource('src/shaders/gravity_force.wgsl'),
    loadShaderSource('src/shaders/density_inject_impulse.wgsl'),
    loadShaderSource('src/shaders/density_inject_disk.wgsl'),
    loadShaderSource('src/shaders/density_inject_noise.wgsl'),
    loadShaderSource('src/shaders/rotation_compute.wgsl'),
    loadShaderSource('src/shaders/rotation_reduce.wgsl'),
    loadShaderSource('src/shaders/rotation_energy.wgsl'),
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
      noise:                noiseLattice + velocityNoise,
      render:               velocityRender,
      densityAdvect,
      gravityPoisson,
      gravityForce,
      densityInjectImpulse,
      densityInjectDisk,
      densityInjectNoise:   noiseLattice + densityInjectNoise,
    },
    rotation: {
      compute: rotationCompute,
      reduce:  rotationReduce,
      energy:  rotationEnergy,
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
  const fieldsContainer = document.getElementById('fields');
  fieldsContainer.style.gap = `${DISPLAY_GAP}px`;
  fieldsContainer.classList.toggle('fields-stacked', FIELD_LAYOUT.stacked);

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
    shaders.rotation.energy,
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
  new ControlHintDisplay(
    document.getElementById('page-description'),
    document.getElementById('control-hint'),
  );
  blockSelectTypeAhead();

  let animationFrameId = null;
  let paused           = false;

  // Reference physics timestep — one frame at 60 fps.
  const PHYSICS_DT = 1 / 60;

  function renderFields() {
    focusMask.syncRadiusWithBrush();
    fluidField.render();
    rotationField.recomputeFrom(fluidField.velocityTexture, fluidField.stepGeneration);
    focusMask.updateOverlay();
    snapshotSaver.saveIfRequested();
  }

  const snapshotSaver = new FieldSnapshotSaver(canvasMain, canvasRotation, () => {
    if (paused) renderFields();
  });

  function queueStartPattern() {
    patternInjector.queueInjectionSeries(START_PATTERN_CENTER, START_PULSES.count, START_PULSES.intervalSeconds);
  }

  function applyRandomSceneSettings() {
    randomizeSceneSettings();
    controlPanel.refreshFromConfig();
    patternInjector.refreshFromConfig();
    history.replaceState(null, '', buildShareUrl());
  }

  function restartWithRandomScene() {
    applyRandomSceneSettings();
    fluidField.reset();
    queueStartPattern();
    if (paused) renderFields();
  }

  new RandomSceneShortcut(restartWithRandomScene);
  if (OPENED_FROM_CANONICAL_LINK) applyRandomSceneSettings();
  queueStartPattern();

  function renderFrame() {
    // Apply input once per display frame — injections are not sub-stepped.
    mouseInjector.applyPendingInjection();
    patternInjector.applyPendingPattern();

    const speed = PHYSICS_DEFAULTS.simulationSpeed;
    if (speed <= 1) {
      // Slow-motion: one step per frame with proportional dt — renders smoothly.
      // Physics regime shifts slightly (pressure converges better, vorticity scales),
      // but this is imperceptible versus the alternative of skipping frames entirely.
      fluidField.step(PHYSICS_DT * speed);
    } else {
      // Fast-forward: multiple fixed-dt sub-steps per frame — preserves physics regime.
      const steps = Math.round(speed);
      for (let i = 0; i < steps; i++) fluidField.step(PHYSICS_DT);
    }

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
  document.getElementById('clear-dynamics-button').addEventListener('click', () => fluidField.reset());
  const clearMassButton = document.getElementById('clear-mass-button');
  clearMassButton.hidden = !GRAVITY_EXPERIMENT;
  clearMassButton.addEventListener('click', () => fluidField.clearDensity());
  document.getElementById('reset-button').addEventListener('click', () => {
    window.location.href = buildDefaultSettingsUrl();
  });

  document.getElementById('copy-url-button').addEventListener('click', () => {
    navigator.clipboard.writeText(buildShareUrl());
  });

  document.addEventListener('input', () => {
    focusMask.syncRadiusWithBrush();
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

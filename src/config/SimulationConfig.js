'use strict';

const _p = new URLSearchParams(window.location.search);
const _float = (key, def) => { const v = parseFloat(_p.get(key)); return isFinite(v) ? v : def; };
const _int   = (key, def, min, max) => { const v = parseInt(_p.get(key), 10); return isNaN(v) ? def : Math.max(min, Math.min(max, v)); };
const _str   = (key, def) => _p.has(key) ? _p.get(key) : def;

// All tuneable defaults live here. URL params override them on load.
const DEFAULTS = {
  gridSize:          256,
  dampingLoss:       1e-7, // = 1 - damping; stored as loss for URL precision
  simSpeed:          2.0,
  brushRadius:       2.0,
  brushStrength:     100.0,
  brushSpeed:        1.0,
  patternScale:      0.7,
  pairDistance:       0.02,
  distanceDelta:      0.02,
  pressureIterations: 40,
  vorticity:          0.0,
  velBrightness:      50,   // slider position 0–100
  rotBrightness:      80,   // slider position 0–100
};

const VEL_TONE_BASE     = 30.0;
const ROT_TONE_BASE     = 0.3;

// Reference grid size at which accumulationScale=1 gives calibrated brightness.
// Compensation multiplies scale by (gridSize / ref) × sampleStride² so brightness
// stays constant regardless of grid size or stride.
export const ROTATION_REFERENCE_GRID_SIZE = 100;
export const VEL_LOG_RANGE = 3;
export const ROT_LOG_RANGE = Math.log(ROT_TONE_BASE * Math.exp(3) * 200.0 / ROT_TONE_BASE); 

export const GRID_SIZE = _int('gridSize', DEFAULTS.gridSize, 32, 1024);

export const DISPLAY_SCALE = 7; // pixels per grid cell
export const DISPLAY_GAP = 32; // pixel gap between the two fields

export const PHYSICS_DEFAULTS = {
  viscosity:            0.001,
  damping:              1 - _float('dampingLoss', DEFAULTS.dampingLoss),
  diffusionIterations:  20,
  pressureIterations:   _int('pressureIters', DEFAULTS.pressureIterations, 1, 100),
  simulationSpeed:      _float('simSpeed',     DEFAULTS.simSpeed),
  boundaryMode:         _int('boundary', 0, 0, 2), // 0=wrap 1=absorb 2=reflect
  vorticityStrength:    _float('vorticity', 0.0),
};

// Pixels-per-second at which speedSensitivity=1 yields a 1× strength multiplier.
export const MOUSE_SPEED_REFERENCE = 400;

export const MOUSE_DEFAULTS = {
  impulseRadius:    _float('brushRadius',   DEFAULTS.brushRadius),
  impulseStrength:  _float('brushStrength', DEFAULTS.brushStrength),
  speedSensitivity: _float('brushSpeed',    DEFAULTS.brushSpeed),
  patternScale:     _float('patternScale',  DEFAULTS.patternScale),
};

export const PATTERN_DEFAULTS = {
  pattern: _str('pattern', 'disk-spin'),
};

export const ROTATION_FIELD = {
  parallelThreshold: 0.001,
  accumulationScale: 1.0,
  pairDistance:      _float('pairDistance', DEFAULTS.pairDistance),
  distanceDelta:     _float('distanceDelta', DEFAULTS.distanceDelta),
  sampleStride:      _int('sampleStride', 1, 1, 32),
  maskCenter:        (_p.has('maskCx') && _p.has('maskCy'))
    ? [_float('maskCx', 0), _float('maskCy', 0)]
    : null,
  maskRadius:        _p.has('maskR') ? _float('maskR', 2) : null,
};

const velBrightnessPos = _int('velBrightness', DEFAULTS.velBrightness, 0, 100);
const rotBrightnessPos = _int('rotBrightness', DEFAULTS.rotBrightness, 0, 100);

const brightnessToTone = (base, logRange, pos) =>
  base * Math.exp(logRange * (50 - pos) / 50);

export const RENDER_DEFAULTS = {
  velocityToneMidpoint: brightnessToTone(VEL_TONE_BASE, VEL_LOG_RANGE, velBrightnessPos),
  rotationToneMidpoint: brightnessToTone(ROT_TONE_BASE, ROT_LOG_RANGE, rotBrightnessPos),
};

export const BRIGHTNESS_SLIDER_POSITIONS = {
  velocity: velBrightnessPos,
  rotation: rotBrightnessPos,
};

export function buildShareUrl() {
  const params = new URLSearchParams({
    gridSize:      GRID_SIZE,
    dampingLoss:   (1 - PHYSICS_DEFAULTS.damping).toExponential(3),
    simSpeed:      PHYSICS_DEFAULTS.simulationSpeed.toPrecision(3),
    brushRadius:   MOUSE_DEFAULTS.impulseRadius.toPrecision(3),
    brushStrength: MOUSE_DEFAULTS.impulseStrength.toPrecision(3),
    brushSpeed:    MOUSE_DEFAULTS.speedSensitivity.toPrecision(3),
    patternScale:  MOUSE_DEFAULTS.patternScale.toPrecision(3),
    pairDistance:  ROTATION_FIELD.pairDistance.toPrecision(3),
    distanceDelta: ROTATION_FIELD.distanceDelta.toPrecision(3),
    sampleStride:  ROTATION_FIELD.sampleStride,
    boundary:      PHYSICS_DEFAULTS.boundaryMode,
    vorticity:      PHYSICS_DEFAULTS.vorticityStrength.toPrecision(3),
    pressureIters:  PHYSICS_DEFAULTS.pressureIterations,
    pattern:       PATTERN_DEFAULTS.pattern,
    velBrightness: BRIGHTNESS_SLIDER_POSITIONS.velocity,
    rotBrightness: BRIGHTNESS_SLIDER_POSITIONS.rotation,
  });
  if (ROTATION_FIELD.maskCenter) {
    params.set('maskCx', ROTATION_FIELD.maskCenter[0].toPrecision(5));
    params.set('maskCy', ROTATION_FIELD.maskCenter[1].toPrecision(5));
    params.set('maskR',  ROTATION_FIELD.maskRadius.toPrecision(4));
  }
  return `${window.location.origin}${window.location.pathname}?${params}`;
}

export const COLORS = {
  rotationPositive: [1.0, 0.5, 0.0], // orange — counter-clockwise
  rotationNegative: [0.0, 0.6, 1.0], // blue  — clockwise
  rotationZero: [0.0, 0.0, 0.0], // black — no contribution
};

// Named physics model identifiers used by PhysicsRegistry.
export const PHYSICS_MODELS = {
  NAVIER_STOKES: 'navier-stokes',
};

export const DEFAULT_PHYSICS_MODEL = PHYSICS_MODELS.NAVIER_STOKES;

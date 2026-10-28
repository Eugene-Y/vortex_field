'use strict';

const _p = new URLSearchParams(window.location.search);
const _float = (key, def) => { const v = parseFloat(_p.get(key)); return isFinite(v) ? v : def; };
const _int   = (key, def, min, max) => { const v = parseInt(_p.get(key), 10); return isNaN(v) ? def : Math.max(min, Math.min(max, v)); };
const _str   = (key, def) => _p.has(key) ? _p.get(key) : def;

const HEX_COLOR_PATTERN = /^[0-9a-f]{6}$/i;
const COLOR_CHANNEL_MAX = 255;

// Colors are [r, g, b] floats in 0–1 internally and 6-digit hex (no '#') in URLs and color inputs.
export const rgbToHex = rgb => rgb
  .map(channel => Math.round(channel * COLOR_CHANNEL_MAX).toString(16).padStart(2, '0'))
  .join('');
export const hexToRgb = hex => [0, 2, 4]
  .map(offset => parseInt(hex.slice(offset, offset + 2), 16) / COLOR_CHANNEL_MAX);
const _color = (key, def) => { const v = _p.get(key); return v && HEX_COLOR_PATTERN.test(v) ? hexToRgb(v) : def; };

// All tuneable defaults live here. URL params override them on load.
const DEFAULTS = {
  gridSize:          512,
  dampingLoss:       1e-7, // = 1 - damping; stored as loss for URL precision
  simSpeed:          1.0,
  brushRadius:       2.0,
  brushStrength:     100.0,
  brushSpeed:        1.0,
  patternScale:      0.7,
  pairDistance:       0.0447,
  distanceDelta:      0.00302,
  pressureIterations: 75,
  vorticity:          0.605,
  velBrightness:      46,   // slider position 0–100
  rotBrightness:      55,   // slider position 0–100; 50 = midpoint = 1× normalized mean
  densBrightness:     50,   // slider position 0–100
};

const VEL_TONE_BASE  = 30.0;
// After energy normalization the mean active cell has value ~1.0, so the tone
// midpoint is expressed relative to that unit-mean scale.
const ROT_TONE_BASE  = 1.0;
// Density values accumulate from injection; midpoint ~10 gives good default visibility.
const DENS_TONE_BASE = 10.0;

// Reference grid size at which accumulationScale=1 gives calibrated brightness.
// Compensation multiplies scale by (gridSize / ref) × sampleStride² so brightness
// stays constant regardless of grid size or stride.
export const ROTATION_REFERENCE_GRID_SIZE = 100;
export const VEL_LOG_RANGE  = 3;
export const DENS_LOG_RANGE = 3;
// Asymmetric log range for the rotation brightness slider.
// Dim side is compressed (less dead travel in the near-black range).
// Bright side is extended so the slider reaches 100× brighter than the old symmetric limit.
export const ROT_DIM_LOG_RANGE    = 3.5;                                        // exp(3.5) ≈ 33  at pos=0
export const ROT_BRIGHT_LOG_RANGE = Math.log(Math.exp(3) * 200) + Math.log(100); // ≈ 12.2 at pos=100

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
  vorticityStrength:    _float('vorticity', DEFAULTS.vorticity),
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

// The mass/gravity experiment is hidden and inert unless the URL opts in with ?gravityExperiment=1,
// so a stray gravEnabled/brushMode in a shared link cannot switch it on behind hidden controls.
export const GRAVITY_EXPERIMENT = _int('gravityExperiment', 0, 0, 1) === 1;

export const BRUSH = {
  mode: GRAVITY_EXPERIMENT ? _str('brushMode', 'impulse') : 'impulse',  // 'impulse' | 'mass'
};

export const GRAVITY = {
  enabled:        GRAVITY_EXPERIMENT && _int('gravEnabled', 0, 0, 1) === 1,
  strength:       _float('gravStrength', 5.0),
  iterations:     _int('gravIters', 20, 1, 100),
  densityDamping: _float('densDamping', 0.999),
};

export const ROTATION_FIELD = {
  parallelThreshold: 0.001,
  accumulationScale: 1.0,
  pairDistance:      _float('pairDistance', DEFAULTS.pairDistance),
  distanceDelta:     _float('distanceDelta', DEFAULTS.distanceDelta),
  showMask:          _int('showMask', 1, 0, 3),
  sampleStride:      _int('sampleStride', 1, 1, 32),
  autoNormalize:     _int('autoNorm', 1, 0, 1),
  maskCenter:        (_p.has('maskCx') && _p.has('maskCy'))
    ? [_float('maskCx', 0), _float('maskCy', 0)]
    : null,
  maskRadius:        _p.has('maskR') ? _float('maskR', 2) : null,
};

const velBrightnessPos  = _int('velBrightness',  DEFAULTS.velBrightness,  0, 100);
const rotBrightnessPos  = _int('rotBrightness',  DEFAULTS.rotBrightness,  0, 100);
const densBrightnessPos = _int('densBrightness', DEFAULTS.densBrightness, 0, 100);

// t = +1 at pos=0 (dim end), -1 at pos=100 (bright end).
// dimLogRange and brightLogRange may differ for an asymmetric slider.
const brightnessToTone = (base, dimLogRange, brightLogRange, pos) => {
  const t = (50 - pos) / 50;
  return base * Math.exp((t >= 0 ? dimLogRange : brightLogRange) * t);
};

export const RENDER_DEFAULTS = {
  velocityToneMidpoint: brightnessToTone(VEL_TONE_BASE,  VEL_LOG_RANGE,      VEL_LOG_RANGE,        velBrightnessPos),
  rotationToneMidpoint: brightnessToTone(ROT_TONE_BASE,  ROT_DIM_LOG_RANGE,  ROT_BRIGHT_LOG_RANGE, rotBrightnessPos),
  densityToneMidpoint:  brightnessToTone(DENS_TONE_BASE, DENS_LOG_RANGE,     DENS_LOG_RANGE,       densBrightnessPos),
};

export const BRIGHTNESS_SLIDER_POSITIONS = {
  velocity: velBrightnessPos,
  rotation: rotBrightnessPos,
  density:  densBrightnessPos,
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
    showMask:      ROTATION_FIELD.showMask,
    sampleStride:  ROTATION_FIELD.sampleStride,
    autoNorm:      ROTATION_FIELD.autoNormalize,
    boundary:      PHYSICS_DEFAULTS.boundaryMode,
    vorticity:     PHYSICS_DEFAULTS.vorticityStrength.toPrecision(3),
    pressureIters: PHYSICS_DEFAULTS.pressureIterations,
    pattern:       PATTERN_DEFAULTS.pattern,
    velBrightness: BRIGHTNESS_SLIDER_POSITIONS.velocity,
    rotBrightness: BRIGHTNESS_SLIDER_POSITIONS.rotation,
    colorCcw:      rgbToHex(COLORS.rotationPositive),
    colorCw:       rgbToHex(COLORS.rotationNegative),
    colorZero:     rgbToHex(COLORS.rotationZero),
  });
  if (GRAVITY_EXPERIMENT) {
    params.set('gravityExperiment', 1);
    params.set('brushMode',      BRUSH.mode);
    params.set('gravEnabled',    GRAVITY.enabled ? 1 : 0);
    params.set('gravStrength',   GRAVITY.strength.toPrecision(3));
    params.set('gravIters',      GRAVITY.iterations);
    params.set('densDamping',    GRAVITY.densityDamping.toPrecision(4));
    params.set('densBrightness', BRIGHTNESS_SLIDER_POSITIONS.density);
  }
  if (ROTATION_FIELD.maskCenter) {
    params.set('maskCx', ROTATION_FIELD.maskCenter[0].toPrecision(5));
    params.set('maskCy', ROTATION_FIELD.maskCenter[1].toPrecision(5));
    params.set('maskR',  ROTATION_FIELD.maskRadius.toPrecision(4));
  }
  return `${window.location.origin}${window.location.pathname}?${params}`;
}

// Mutable: the Field B color pickers replace these arrays live.
export const COLORS = {
  rotationPositive: _color('colorCcw',  [1.0, 0.5, 0.0]), // orange — counter-clockwise
  rotationNegative: _color('colorCw',   [0.0, 0.6, 1.0]), // blue   — clockwise
  rotationZero:     _color('colorZero', [0.0, 0.0, 0.0]), // black  — no contribution
};

// Named physics model identifiers used by PhysicsRegistry.
export const PHYSICS_MODELS = {
  NAVIER_STOKES: 'navier-stokes',
};

export const DEFAULT_PHYSICS_MODEL = PHYSICS_MODELS.NAVIER_STOKES;

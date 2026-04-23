'use strict';

const _p = new URLSearchParams(window.location.search);
const _float = (key, def) => { const v = parseFloat(_p.get(key)); return isFinite(v) ? v : def; };
const _int   = (key, def, min, max) => { const v = parseInt(_p.get(key), 10); return isNaN(v) ? def : Math.max(min, Math.min(max, v)); };

// All tuneable defaults live here. URL params override them on load.
const DEFAULTS = {
  gridSize:      64,
  damping:       0.995,
  simSpeed:      1.0,
  brushRadius:   2.0,
  brushStrength: 100.0,
  patternScale:  0.5,
  pairRange:     1.0,
  velBrightness: 30.0,
  rotBrightness: 0.3,
};

export const GRID_SIZE = _int('gridSize', DEFAULTS.gridSize, 32, 512);

export const DISPLAY_SCALE = 8; // pixels per grid cell
export const DISPLAY_GAP = 32; // pixel gap between the two fields

export const PHYSICS_DEFAULTS = {
  viscosity:            0.001,
  damping:              _float('damping',      DEFAULTS.damping),
  diffusionIterations:  20,
  pressureIterations:   40,
  simulationSpeed:      _float('simSpeed',     DEFAULTS.simSpeed),
};

export const MOUSE_DEFAULTS = {
  impulseRadius:   _float('brushRadius',   DEFAULTS.brushRadius),
  impulseStrength: _float('brushStrength', DEFAULTS.brushStrength),
  patternScale:    _float('patternScale',  DEFAULTS.patternScale),
};

export const ROTATION_FIELD = {
  parallelThreshold: 0.001,
  accumulationScale: 1.0,
  pairRange:         _float('pairRange', DEFAULTS.pairRange),
};

export const RENDER_DEFAULTS = {
  velocityToneMidpoint: _float('velBrightness', DEFAULTS.velBrightness),
  rotationToneMidpoint: _float('rotBrightness', DEFAULTS.rotBrightness),
};

export const RENDER_BRIGHTNESS_DEFAULTS = {
  velocityToneMidpoint: DEFAULTS.velBrightness,
  rotationToneMidpoint: DEFAULTS.rotBrightness,
};

export function buildShareUrl() {
  const params = new URLSearchParams({
    gridSize:      GRID_SIZE,
    damping:       PHYSICS_DEFAULTS.damping.toPrecision(4),
    simSpeed:      PHYSICS_DEFAULTS.simulationSpeed.toPrecision(3),
    brushRadius:   MOUSE_DEFAULTS.impulseRadius.toPrecision(3),
    brushStrength: MOUSE_DEFAULTS.impulseStrength.toPrecision(3),
    patternScale:  MOUSE_DEFAULTS.patternScale.toPrecision(3),
    pairRange:     ROTATION_FIELD.pairRange.toPrecision(3),
    velBrightness: RENDER_DEFAULTS.velocityToneMidpoint.toPrecision(3),
    rotBrightness: RENDER_DEFAULTS.rotationToneMidpoint.toPrecision(3),
  });
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

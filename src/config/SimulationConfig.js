'use strict';

export const GRID_SIZE = 64;

export const DISPLAY_SCALE = 8; // pixels per grid cell

export const PHYSICS_DEFAULTS = {
  viscosity: 0.1,
  damping: 0.995,
  diffusionIterations: 20,
  pressureIterations: 40,
};

export const MOUSE_DEFAULTS = {
  impulseRadius: 5.0,
  impulseStrength: 200.0,
};

export const ROTATION_FIELD = {
  // Below this cross-product magnitude the pair has no meaningful rotation center.
  // Numerically: |v_i × v_j| / (|v_i| * |v_j|) < threshold → discard.
  parallelThreshold: 0.001,
  accumulationScale: 1.0,
};

export const COLORS = {
  rotationPositive: [1.0, 0.5, 0.0], // orange — counter-clockwise
  rotationNegative: [0.0, 0.6, 1.0], // blue  — clockwise
  rotationZero:     [0.0, 0.0, 0.0], // black — no contribution
};

// Named physics model identifiers used by PhysicsRegistry.
export const PHYSICS_MODELS = {
  NAVIER_STOKES: 'navier-stokes',
};

export const DEFAULT_PHYSICS_MODEL = PHYSICS_MODELS.NAVIER_STOKES;

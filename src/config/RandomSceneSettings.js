'use strict';

import { DEFAULTS, BRUSH_STRENGTH_MAX, PHYSICS_DEFAULTS, MOUSE_DEFAULTS, PATTERN_DEFAULTS, START_PULSES, ROTATION_FIELD, COLORS } from './SimulationConfig.js';

const SHOW_CCW_ONLY = 1;
const SHOW_CW_ONLY  = 2;
const SHOW_BOTH     = 3;

const ORANGE_BLUE_PALETTE = {
  positive: DEFAULTS.colorCcw,
  zero:     DEFAULTS.colorZero,
  negative: DEFAULTS.colorCw,
};
const BLACK_WHITE_PALETTE = {
  positive: [0.0, 0.0, 0.0],
  zero:     [0.5, 0.5, 0.5],
  negative: [1.0, 1.0, 1.0],
};

const START_PATTERNS = ['disk-spin', 'disk-explode', 'circle'];

const BOUNDARY_WRAP    = 0;
const BOUNDARY_ABSORB  = 1;
const BOUNDARY_REFLECT = 2;

const SIMULATION_SPEED_MIN = 0.5;
const SIMULATION_SPEED_MAX = 5.0;
const SLOW_MOTION_LIMIT    = 1.0;

const VORTICITY_MAX = 2.0;

const PRESSURE_ITERATIONS_MIN = 1;
const PRESSURE_ITERATIONS_MAX = 100;
const INCOMPRESSIBILITY_DIVISIONS = 16;

const PATTERN_SCALE_MIN = 0.1;
const PATTERN_SCALE_MAX = 1.0;

const START_PULSES_MAX           = 4;
const START_INTERVAL_MIN_SECONDS = 0.125;
const START_INTERVAL_MAX_SECONDS = 0.5;

const PAIR_DISTANCE_MIN = 0.003;
const PAIR_DISTANCE_MAX = 1.0;

const randomUniform    = (min, max) => min + (max - min) * Math.random();
const randomLogUniform = (min, max) => Math.exp(randomUniform(Math.log(min), Math.log(max)));
const randomInteger    = (min, max) => min + Math.floor(Math.random() * (max - min + 1));
const randomChoice     = choices => choices[randomInteger(0, choices.length - 1)];

// Half the draws are slow motion and half fast-forward; log-uniform within each half, like the slider.
function randomSimulationSpeed() {
  return Math.random() < 0.5
    ? randomLogUniform(SIMULATION_SPEED_MIN, SLOW_MOTION_LIMIT)
    : randomLogUniform(SLOW_MOTION_LIMIT, SIMULATION_SPEED_MAX);
}

function randomPressureIterations() {
  const division = randomInteger(0, INCOMPRESSIBILITY_DIVISIONS);
  const span     = PRESSURE_ITERATIONS_MAX - PRESSURE_ITERATIONS_MIN;
  return Math.round(PRESSURE_ITERATIONS_MIN + span * division / INCOMPRESSIBILITY_DIVISIONS);
}

function applyRotationPalette(palette) {
  COLORS.rotationPositive = [...palette.positive];
  COLORS.rotationZero     = [...palette.zero];
  COLORS.rotationNegative = [...palette.negative];
}

/** Writes a random starting scene into the live config; settings not listed here keep their values. */
export function randomizeSceneSettings() {
  ROTATION_FIELD.showMask = randomChoice([SHOW_CCW_ONLY, SHOW_CW_ONLY, SHOW_BOTH]);
  applyRotationPalette(randomChoice([ORANGE_BLUE_PALETTE, BLACK_WHITE_PALETTE]));
  PATTERN_DEFAULTS.pattern = randomChoice(START_PATTERNS);
  PHYSICS_DEFAULTS.boundaryMode       = randomChoice([BOUNDARY_WRAP, BOUNDARY_ABSORB, BOUNDARY_REFLECT]);
  PHYSICS_DEFAULTS.simulationSpeed    = randomSimulationSpeed();
  PHYSICS_DEFAULTS.vorticityStrength  = randomUniform(0, VORTICITY_MAX);
  PHYSICS_DEFAULTS.pressureIterations = randomPressureIterations();
  MOUSE_DEFAULTS.impulseStrength = randomLogUniform(DEFAULTS.brushStrength, BRUSH_STRENGTH_MAX);
  MOUSE_DEFAULTS.patternScale    = randomUniform(PATTERN_SCALE_MIN, PATTERN_SCALE_MAX);
  START_PULSES.count           = randomInteger(1, START_PULSES_MAX);
  START_PULSES.intervalSeconds = randomUniform(START_INTERVAL_MIN_SECONDS, START_INTERVAL_MAX_SECONDS);
  ROTATION_FIELD.pairDistance = randomLogUniform(PAIR_DISTANCE_MIN, PAIR_DISTANCE_MAX);
}

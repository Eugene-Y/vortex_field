'use strict';

// Synthetic velocity fields (cells/s, rg32float layout, row 0 = bottom) that bracket
// the Field B workload: coherent vortices (contention on vortex cores), one solid-body
// vortex (every pair shares the same center — worst-case atomic contention), and white
// noise (centers scattered uniformly — no contention, worst-case cache behaviour).

const PEAK_SPEED            = 60;
const VORTEX_COUNT          = 14;
const VORTEX_CORE_MIN       = 3;
const VORTEX_CORE_MAX       = 18;
const SOLID_BODY_RADIUS     = 0.4;
const TURBULENT_NOISE_SHARE = 0.15;
const RANDOM_SEED           = 12345;

function createSeededRandom(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let mixed = Math.imul(state ^ (state >>> 15), 1 | state);
    mixed ^= mixed + Math.imul(mixed ^ (mixed >>> 7), 61 | mixed);
    return ((mixed ^ (mixed >>> 14)) >>> 0) / 4294967296;
  };
}

function shortestPeriodicDelta(delta, gridSize) {
  return delta - gridSize * Math.round(delta / gridSize);
}

// Lamb–Oseen profile: solid-body core, 1/r far field, peak speed at r ≈ 1.12 · core.
function addLambOseenVortex(velocity, gridSize, centerX, centerY, coreRadius, peakSpeed) {
  const circulationScale = peakSpeed * coreRadius / 0.638;
  for (let row = 0; row < gridSize; row++) {
    for (let col = 0; col < gridSize; col++) {
      const dx = shortestPeriodicDelta(col - centerX, gridSize);
      const dy = shortestPeriodicDelta(row - centerY, gridSize);
      const radiusSquared = dx * dx + dy * dy;
      if (radiusSquared < 1e-6) continue;
      const tangentialSpeed = circulationScale * (1 - Math.exp(-radiusSquared / (coreRadius * coreRadius)))
        / Math.sqrt(radiusSquared);
      const radius = Math.sqrt(radiusSquared);
      const index  = 2 * (row * gridSize + col);
      velocity[index]     += -dy / radius * tangentialSpeed;
      velocity[index + 1] +=  dx / radius * tangentialSpeed;
    }
  }
}

function buildVortexField(gridSize, random) {
  const velocity = new Float32Array(gridSize * gridSize * 2);
  for (let vortex = 0; vortex < VORTEX_COUNT; vortex++) {
    const coreRadius = (VORTEX_CORE_MIN + random() * (VORTEX_CORE_MAX - VORTEX_CORE_MIN)) * gridSize / 256;
    const direction  = random() < 0.5 ? -1 : 1;
    addLambOseenVortex(velocity, gridSize, random() * gridSize, random() * gridSize,
      coreRadius, direction * PEAK_SPEED * (0.4 + 0.6 * random()));
  }
  return velocity;
}

function buildSolidBodyRotation(gridSize) {
  const velocity  = new Float32Array(gridSize * gridSize * 2);
  // Off the cell corners, so the shared center does not straddle four cells by rounding.
  const center    = gridSize / 2 + 0.37;
  const maxRadius = SOLID_BODY_RADIUS * gridSize;
  const angularVelocity = PEAK_SPEED / maxRadius;
  for (let row = 0; row < gridSize; row++) {
    for (let col = 0; col < gridSize; col++) {
      const dx = col - center;
      const dy = row - center;
      if (dx * dx + dy * dy > maxRadius * maxRadius) continue;
      const index = 2 * (row * gridSize + col);
      velocity[index]     = -dy * angularVelocity;
      velocity[index + 1] =  dx * angularVelocity;
    }
  }
  return velocity;
}

function buildWhiteNoise(gridSize, random) {
  const velocity = new Float32Array(gridSize * gridSize * 2);
  for (let index = 0; index < velocity.length; index++) {
    velocity[index] = (random() * 2 - 1) * PEAK_SPEED;
  }
  return velocity;
}

function buildTurbulentField(gridSize, random) {
  const velocity = buildVortexField(gridSize, random);
  for (let index = 0; index < velocity.length; index++) {
    velocity[index] += (random() * 2 - 1) * PEAK_SPEED * TURBULENT_NOISE_SHARE;
  }
  return velocity;
}

export const VELOCITY_SCENARIOS = {
  vortices:   gridSize => buildVortexField(gridSize, createSeededRandom(RANDOM_SEED)),
  turbulent:  gridSize => buildTurbulentField(gridSize, createSeededRandom(RANDOM_SEED)),
  solidBody:  gridSize => buildSolidBodyRotation(gridSize),
  whiteNoise: gridSize => buildWhiteNoise(gridSize, createSeededRandom(RANDOM_SEED)),
};

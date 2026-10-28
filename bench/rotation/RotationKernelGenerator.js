'use strict';

// Generates experimental Field B accumulation kernels from orthogonal options so each
// optimization can be measured alone and in combination against the production shader.
//
// Options:
//   pairMath:     'reference' — the production center/ω computation, line for line
//                 'inverseT'  — closed form ω = 1/t (see buildInverseTPairMath)
//   offsetTable:  'full'      — production table, pairs deduplicated by indexB > indexA
//                 'halfPlane' — only offsets with dRow > 0 || (dRow == 0 && dCol > 0)
//   accumulation: 'casF32' | 'fixed32' | 'fixed64' | 'none'
//   aggregation:  'none' | 'subgroup' | 'workgroup' — merge same-target contributions
//                 before the storage atomic (workgroup implies fixed64 accumulation)
//   aggregationRounds: number of leader rounds for subgroup aggregation
//   offsetSplit:  1..n — dispatch dimension y splits the offset table into n chunks

const PARAMS_STRUCT = /* wgsl */`
struct Params {
  gridSize:          u32,
  accumulationScale: f32,
  parallelThreshold: f32,
  offsetCount:       u32,
  minRadius:         f32,
  maxRadius:         f32,
  sampleStride:      u32,
  maskOriginX:       u32,
  maskOriginY:       u32,
  maskBoxSize:       u32,
  maskCenterX:       f32,
  maskCenterY:       f32,
  maskRadiusSq:      f32,
  useMask:           u32,
  boundaryMode:      u32,
  showMask:          u32,
}

struct PairContribution {
  valid:       bool,
  targetIndex: u32,
  value:       f32,
}

@group(0) @binding(0) var velocityTexture: texture_2d<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read> annulusOffsets: array<i32>;

override ACCUMULATION_BUFFERS: u32 = 16u;
override FIXED_POINT_SCALE: f32 = 4294967296.0;

const INVALID_TARGET = 0xFFFFFFFFu;
const NO_CONTRIBUTION = PairContribution(false, 0u, 0.0);

fn insideMaskCircle(col: i32, row: i32) -> bool {
  let dc = f32(col) - params.maskCenterX;
  let dr = f32(row) - params.maskCenterY;
  return dc * dc + dr * dr <= params.maskRadiusSq;
}

fn passesSignFilter(value: f32) -> bool {
  if ((params.showMask & 1u) == 0u && value > 0.0) { return false; }
  if ((params.showMask & 2u) == 0u && value < 0.0) { return false; }
  return true;
}

fn targetIndexOfCenter(center: vec2f) -> u32 {
  let gridSize = i32(params.gridSize);
  let centerCol = i32(center.x) % gridSize;
  let centerRow = i32(center.y) % gridSize;
  return u32(centerRow) * params.gridSize + u32(centerCol);
}
`;

const REFERENCE_PAIR_MATH = /* wgsl */`
fn periodicDisplacement(posA: vec2f, posB: vec2f, gridSize: f32) -> vec2f {
  var delta = posB - posA;
  delta -= gridSize * floor((delta + 0.5 * gridSize) / gridSize);
  return delta;
}

fn pairDisplacement(posA: vec2f, posB: vec2f, gridSize: f32) -> vec2f {
  if (params.boundaryMode == 0u) { return periodicDisplacement(posA, posB, gridSize); }
  return posB - posA;
}

fn computeInstantaneousRotationCenter(posA: vec2f, velA: vec2f, posB: vec2f, velB: vec2f, gridSize: f32) -> vec2f {
  let invalid = vec2f(1e38, 1e38);
  let lenA = length(velA);
  let lenB = length(velB);
  if (lenA < 1e-6 || lenB < 1e-6) { return invalid; }
  let normalA = vec2f(-velA.y, velA.x);
  let normalB = vec2f(-velB.y, velB.x);
  let denominator = normalA.x * normalB.y - normalA.y * normalB.x;
  if (abs(denominator) / (lenA * lenB) < params.parallelThreshold) { return invalid; }
  let delta = pairDisplacement(posA, posB, gridSize);
  let t     = (delta.x * normalB.y - delta.y * normalB.x) / denominator;
  let rawCenter = posA + t * normalA;
  if (params.boundaryMode == 0u) { return rawCenter - gridSize * floor(rawCenter / gridSize); }
  return rawCenter;
}

fn computeAngularVelocity(velA: vec2f, center: vec2f, posA: vec2f, gridSize: f32) -> f32 {
  let arm = pairDisplacement(center, posA, gridSize);
  let armLengthSquared = dot(arm, arm);
  if (armLengthSquared < 0.01) { return 0.0; }
  return (arm.x * velA.y - arm.y * velA.x) / armLengthSquared;
}

fn evaluatePair(posA: vec2f, velA: vec2f, offset: vec2i, posB: vec2f, velB: vec2f) -> PairContribution {
  let gSize  = f32(params.gridSize);
  let center = computeInstantaneousRotationCenter(posA, velA, posB, velB, gSize);
  if (center.x > 1e37) { return NO_CONTRIBUTION; }
  if (params.boundaryMode != 0u &&
      (center.x < 0.0 || center.x >= gSize || center.y < 0.0 || center.y >= gSize)) { return NO_CONTRIBUTION; }
  let armB = pairDisplacement(center, posB, gSize);
  if (dot(armB, armB) < 0.01) { return NO_CONTRIBUTION; }
  let omegaA = computeAngularVelocity(velA, center, posA, gSize);
  let omegaB = computeAngularVelocity(velB, center, posB, gSize);
  let value = (omegaA + omegaB) * 0.5 * params.accumulationScale / (gSize * gSize);
  if (!passesSignFilter(value)) { return NO_CONTRIBUTION; }
  return PairContribution(true, targetIndexOfCenter(center), value);
}
`;

// The center lies on both normals: c = posA + t·perp(velA) = posB + s·perp(velB).
// Solving with Cramer's rule, every cross product with a perpendicular collapses to a dot:
//   t = dot(delta, velB) / cross(velA, velB),   s = dot(delta, velA) / cross(velA, velB)
// and the arm from c to A is −t·perp(velA), so ω_A = cross(arm, velA) / |arm|² = 1/t
// (angular velocity = speed / radius), likewise ω_B = 1/s.
// Equal to the reference except when the center is more than gridSize/2 from a cell in
// Wrap mode: there the reference measures the arm to the nearest torus image of the center.
const INVERSE_T_PAIR_MATH = /* wgsl */`
const MIN_SPEED_SQUARED     = 1e-12;
const MIN_ARM_LENGTH_SQUARED = 0.01;

fn crossProduct(a: vec2f, b: vec2f) -> f32 { return a.x * b.y - a.y * b.x; }

fn evaluatePair(posA: vec2f, velA: vec2f, offset: vec2i, posB: vec2f, velB: vec2f) -> PairContribution {
  let speedSquaredA = dot(velA, velA);
  let speedSquaredB = dot(velB, velB);
  if (speedSquaredB < MIN_SPEED_SQUARED) { return NO_CONTRIBUTION; }

  let velocityCross = crossProduct(velA, velB);
  let thresholdSquared = params.parallelThreshold * params.parallelThreshold;
  if (velocityCross * velocityCross < thresholdSquared * speedSquaredA * speedSquaredB) { return NO_CONTRIBUTION; }

  let delta = vec2f(offset);
  let projectionB = dot(delta, velB);
  let projectionA = dot(delta, velA);
  let inverseCross = 1.0 / velocityCross;
  let parameterA = projectionB * inverseCross;
  let parameterB = projectionA * inverseCross;

  let gSize = f32(params.gridSize);
  var center = posA + parameterA * vec2f(-velA.y, velA.x);
  if (params.boundaryMode == 0u) {
    center -= gSize * floor(center / gSize);
  } else if (center.x < 0.0 || center.x >= gSize || center.y < 0.0 || center.y >= gSize) {
    return NO_CONTRIBUTION;
  }

  if (parameterB * parameterB * speedSquaredB < MIN_ARM_LENGTH_SQUARED) { return NO_CONTRIBUTION; }
  let omegaA = select(1.0 / parameterA, 0.0, parameterA * parameterA * speedSquaredA < MIN_ARM_LENGTH_SQUARED);
  let omegaB = 1.0 / parameterB;
  let value = (omegaA + omegaB) * 0.5 * params.accumulationScale / (gSize * gSize);
  if (!passesSignFilter(value)) { return NO_CONTRIBUTION; }
  return PairContribution(true, targetIndexOfCenter(center), value);
}
`;

// Same closed-form center as INVERSE_T_PAIR_MATH, but in Wrap mode the arms are folded
// to the nearest torus image before computing ω, reproducing the production semantics
// exactly (there ω is measured from the folded center, not along the normal).
const INVERSE_T_EXACT_PAIR_MATH = /* wgsl */`
const MIN_SPEED_SQUARED      = 1e-12;
const MIN_ARM_LENGTH_SQUARED = 0.01;

fn crossProduct(a: vec2f, b: vec2f) -> f32 { return a.x * b.y - a.y * b.x; }

fn nearestTorusImage(vector: vec2f, gridSize: f32) -> vec2f {
  return vector - gridSize * floor((vector + 0.5 * gridSize) / gridSize);
}

fn angularVelocityAboutArm(arm: vec2f, velocity: vec2f) -> f32 {
  return crossProduct(arm, velocity) / dot(arm, arm);
}

fn evaluatePair(posA: vec2f, velA: vec2f, offset: vec2i, posB: vec2f, velB: vec2f) -> PairContribution {
  let speedSquaredA = dot(velA, velA);
  let speedSquaredB = dot(velB, velB);
  if (speedSquaredB < MIN_SPEED_SQUARED) { return NO_CONTRIBUTION; }

  let velocityCross = crossProduct(velA, velB);
  let thresholdSquared = params.parallelThreshold * params.parallelThreshold;
  if (velocityCross * velocityCross < thresholdSquared * speedSquaredA * speedSquaredB) { return NO_CONTRIBUTION; }

  let delta = vec2f(offset);
  let inverseCross = 1.0 / velocityCross;
  let parameterA = dot(delta, velB) * inverseCross;
  let parameterB = dot(delta, velA) * inverseCross;

  let gSize = f32(params.gridSize);
  var center = posA + parameterA * vec2f(-velA.y, velA.x);
  var armA = parameterA * vec2f(velA.y, -velA.x);
  var armB = parameterB * vec2f(velB.y, -velB.x);
  if (params.boundaryMode == 0u) {
    center -= gSize * floor(center / gSize);
    armA = nearestTorusImage(armA, gSize);
    armB = nearestTorusImage(armB, gSize);
  } else if (center.x < 0.0 || center.x >= gSize || center.y < 0.0 || center.y >= gSize) {
    return NO_CONTRIBUTION;
  }

  if (dot(armB, armB) < MIN_ARM_LENGTH_SQUARED) { return NO_CONTRIBUTION; }
  let omegaA = select(angularVelocityAboutArm(armA, velA), 0.0, dot(armA, armA) < MIN_ARM_LENGTH_SQUARED);
  let omegaB = angularVelocityAboutArm(armB, velB);
  let value = (omegaA + omegaB) * 0.5 * params.accumulationScale / (gSize * gSize);
  if (!passesSignFilter(value)) { return NO_CONTRIBUTION; }
  return PairContribution(true, targetIndexOfCenter(center), value);
}
`;

const CAS_F32_ACCUMULATION = /* wgsl */`
@group(0) @binding(1) var<storage, read_write> accumulation: array<atomic<i32>>;

fn accumulateValue(index: u32, value: f32) {
  var expected = atomicLoad(&accumulation[index]);
  loop {
    let desired = bitcast<i32>(bitcast<f32>(expected) + value);
    let result  = atomicCompareExchangeWeak(&accumulation[index], expected, desired);
    if (result.exchanged) { break; }
    expected = result.old_value;
  }
}
`;

const FIXED32_ACCUMULATION = /* wgsl */`
@group(0) @binding(1) var<storage, read_write> accumulation: array<atomic<i32>>;

fn accumulateValue(index: u32, value: f32) {
  atomicAdd(&accumulation[index], i32(round(value * FIXED_POINT_SCALE)));
}
`;

// Two's-complement 64-bit add from two 32-bit atomics: the returned previous low word
// tells each invocation whether its own add carried, and addition commutes, so the high
// word converges to the exact sum regardless of interleaving. Small values of either sign
// leave the high increment at zero (−1 + carry 1), so most adds touch one word only.
const FIXED64_ACCUMULATION = /* wgsl */`
@group(0) @binding(1) var<storage, read_write> accumulation: array<atomic<u32>>;

const TWO_POW_32 = 4294967296.0;
const TWO_POW_31 = 2147483648.0;
const FIXED64_LIMIT = 4.6e18;

// Returns (low word, high word) of round(value · FIXED_POINT_SCALE) as two's complement.
fn toFixed64(value: f32) -> vec2u {
  let scaled = clamp(round(value * FIXED_POINT_SCALE), -FIXED64_LIMIT, FIXED64_LIMIT);
  if (abs(scaled) < TWO_POW_31) {
    let narrow = i32(scaled);
    return vec2u(bitcast<u32>(narrow), bitcast<u32>(narrow >> 31u));
  }
  let highPart = floor(scaled / TWO_POW_32);
  return vec2u(u32(scaled - highPart * TWO_POW_32), bitcast<u32>(i32(highPart)));
}

fn addFixed64ToStorage(index: u32, words: vec2u) {
  let previousLow = atomicAdd(&accumulation[2u * index], words.x);
  let carry = select(0u, 1u, previousLow + words.x < previousLow);
  let highIncrement = words.y + carry;
  if (highIncrement != 0u) { atomicAdd(&accumulation[2u * index + 1u], highIncrement); }
}

fn accumulateValue(index: u32, value: f32) {
  addFixed64ToStorage(index, toFixed64(value));
}
`;

// Keeps every pair evaluation alive without touching memory per pair: measures the
// cost of the pair loop alone. The output is meaningless.
const NO_ACCUMULATION = /* wgsl */`
@group(0) @binding(1) var<storage, read_write> accumulation: array<atomic<i32>>;
var<private> sinkSum: f32 = 0.0;

fn accumulateValue(index: u32, value: f32) {
  sinkSum += value + f32(index) * 1e-30;
}

fn flushSink(index: u32) {
  if (sinkSum == 12345.678) { atomicAdd(&accumulation[index], 1); }
}
`;

const ACCUMULATION_SOURCES = {
  casF32:  CAS_F32_ACCUMULATION,
  fixed32: FIXED32_ACCUMULATION,
  fixed64: FIXED64_ACCUMULATION,
  none:    NO_ACCUMULATION,
};

// The lane with the smallest pending target leads; lanes sharing that target are
// summed into one atomic. After the configured rounds, leftovers go straight to memory.
// One round already captures the hot-spot case (a whole subgroup hitting one vortex core)
// while costing a fixed few subgroup ops when targets are scattered.
function buildSubgroupAggregation(rounds) {
  return /* wgsl */`
fn submitContribution(contribution: PairContribution, bufferOffset: u32, laneIndex: u32) {
  var pending = contribution.valid;
  for (var round = 0u; round < ${rounds}u; round++) {
    let candidate = select(INVALID_TARGET, contribution.targetIndex, pending);
    let leaderTarget = subgroupMin(candidate);
    if (leaderTarget == INVALID_TARGET) { return; }
    let joinsLeader = candidate == leaderTarget;
    let leaderSum   = subgroupAdd(select(0.0, contribution.value, joinsLeader));
    let leaderLane  = subgroupMin(select(INVALID_TARGET, laneIndex, joinsLeader));
    if (laneIndex == leaderLane) { accumulateValue(bufferOffset + leaderTarget, leaderSum); }
    pending = pending && !joinsLeader;
  }
  if (pending) { accumulateValue(bufferOffset + contribution.targetIndex, contribution.value); }
}
`;
}

const DIRECT_SUBMISSION = /* wgsl */`
fn submitContribution(contribution: PairContribution, bufferOffset: u32, laneIndex: u32) {
  if (contribution.valid) { accumulateValue(bufferOffset + contribution.targetIndex, contribution.value); }
}
`;

// 'tiled8' maps each 64-thread workgroup onto an 8×8 block of cells instead of a
// 64-cell row segment, so one offset step of a workgroup lands on a compact patch of
// B cells and, in smooth flow, a compact patch of target cells.
function buildCellMapping(cellOrder) {
  if (cellOrder === 'tiled8') {
    return /* wgsl */`
    let tilesPerRow = (params.gridSize + 7u) / 8u;
    let tileIndex   = globalId.x / 64u;
    let laneInTile  = globalId.x % 64u;
    colA = i32((tileIndex % tilesPerRow) * 8u + laneInTile % 8u);
    rowA = i32((tileIndex / tilesPerRow) * 8u + laneInTile / 8u);
    if (colA >= i32(params.gridSize) || rowA >= i32(params.gridSize)) { return; }`;
  }
  return /* wgsl */`
    if (globalId.x >= totalCells) { return; }
    colA = i32(globalId.x % params.gridSize);
    rowA = i32(globalId.x / params.gridSize);`;
}

// 'table' walks the offset list one (dCol, dRow) at a time. 'rowSpans' walks runs of
// consecutive dCol sharing a dRow: the B column just increments, the wrap is one compare,
// and outside Wrap mode the run is clipped to the domain once instead of per pair.
function buildPairLoop(options, submitCall) {
  if (options.offsetTable === 'rowSpans') {
    return /* wgsl */`
  for (var k = firstOffset; k < endOffset; k++) {
    let dRow = annulusOffsets[k * 4u];
    var dColStart = annulusOffsets[k * 4u + 1u];
    var dColEnd   = annulusOffsets[k * 4u + 2u];
    var rowB = rowA + dRow;
    var colB = colA + dColStart;
    if (params.boundaryMode == 0u) {
      rowB += select(select(0, -gridSize, rowB >= gridSize), gridSize, rowB < 0);
      colB += select(select(0, -gridSize, colB >= gridSize), gridSize, colB < 0);
    } else {
      if (rowB < 0 || rowB >= gridSize) { continue; }
      dColStart = max(dColStart, -colA);
      dColEnd   = min(dColEnd, gridSize - 1 - colA);
      colB = colA + dColStart;
    }
    let posBRow = f32(rowB);
    for (var dCol = dColStart; dCol <= dColEnd; dCol++) {
      if (params.useMask == 0u || insideMaskCircle(colB, rowB)) {
        let velB = textureLoad(velocityTexture, vec2i(colB, rowB), 0).xy;
        let contribution = evaluatePair(posA, velA, vec2i(dCol, dRow), vec2f(f32(colB), posBRow), velB);
        ${submitCall}
      }
      colB += 1;
      if (colB == gridSize) { colB = 0; }
    }
  }`;
  }
  const skipsDuplicate = options.offsetTable === 'full' ? 'if (indexB <= indexA) { continue; }' : '';
  return /* wgsl */`
  for (var k = firstOffset; k < endOffset; k++) {
    let offset = vec2i(annulusOffsets[k * 2u], annulusOffsets[k * 2u + 1u]);
    var colB = colA + offset.x;
    var rowB = rowA + offset.y;
    if (params.boundaryMode == 0u) {
      colB = (colB + gridSize) % gridSize;
      rowB = (rowB + gridSize) % gridSize;
    } else if (colB < 0 || colB >= gridSize || rowB < 0 || rowB >= gridSize) {
      continue;
    }
    let indexB = u32(rowB) * params.gridSize + u32(colB);
    ${skipsDuplicate}
    if (params.useMask == 1u && !insideMaskCircle(colB, rowB)) { continue; }

    let velB = textureLoad(velocityTexture, vec2i(colB, rowB), 0).xy;
    let contribution = evaluatePair(posA, velA, offset, vec2f(f32(colB), f32(rowB)), velB);
    ${submitCall}
  }`;
}

function buildEntryPoint(options) {
  const usesSubgroups  = options.aggregation === 'subgroup';
  const laneParameter  = usesSubgroups
    ? '@builtin(subgroup_invocation_id) laneIndex: u32,'
    : '';
  const laneArgument   = usesSubgroups ? 'laneIndex' : '0u';
  const flushSink = options.accumulation === 'none' ? 'flushSink(indexA);' : '';

  return /* wgsl */`
@compute @workgroup_size(64)
fn computeRotation(
  @builtin(global_invocation_id) globalId:    vec3<u32>,
  @builtin(workgroup_id)         workgroupId: vec3<u32>,
  ${laneParameter}
) {
  let totalCells = params.gridSize * params.gridSize;
  var colA: i32;
  var rowA: i32;
  if (params.useMask == 1u) {
    let boxSize = params.maskBoxSize;
    if (globalId.x >= boxSize * boxSize) { return; }
    colA = i32(params.maskOriginX) + i32(globalId.x % boxSize);
    rowA = i32(params.maskOriginY) + i32(globalId.x / boxSize);
    if (!insideMaskCircle(colA, rowA)) { return; }
  } else {
    ${buildCellMapping(options.cellOrder ?? 'rowMajor')}
  }

  let indexA = u32(rowA) * params.gridSize + u32(colA);
  let posA   = vec2f(f32(colA), f32(rowA));
  let velA   = textureLoad(velocityTexture, vec2i(colA, rowA), 0).xy;
  if (length(velA) < 1e-6) { return; }
  if (colA % i32(params.sampleStride) != 0 || rowA % i32(params.sampleStride) != 0) { return; }
  if (params.useMask == 1u && params.minRadius > 2.0 * sqrt(params.maskRadiusSq)) { return; }

  let bufferOffset = (workgroupId.x % ACCUMULATION_BUFFERS) * totalCells;
  let gridSize     = i32(params.gridSize);

  let chunkLength = (params.offsetCount + ${options.offsetSplit}u - 1u) / ${options.offsetSplit}u;
  let firstOffset = globalId.y * chunkLength;
  let endOffset   = min(params.offsetCount, firstOffset + chunkLength);

${buildPairLoop(options, `submitContribution(contribution, bufferOffset, ${laneArgument});`)}
  ${flushSink}
}
`;
}

// A direct-mapped cache of target cells in workgroup memory absorbs hot spots (vortex
// cores collect the centers of most pairs) without subgroup operations: a slot is
// claimed by the first target hashed to it, later contributions to that target add
// there, colliding targets go straight to storage, and the slots are flushed once.
// Exact integer sums make the result independent of which path a contribution takes.
function buildWorkgroupCache(slotCount, workgroupSize) {
  return /* wgsl */`
const CACHE_SLOTS = ${slotCount}u;
const EMPTY_SLOT  = 0xFFFFFFFFu;
const FIBONACCI_HASH = 2654435761u;
const CACHE_HASH_SHIFT = ${32 - Math.log2(slotCount)}u;

var<workgroup> cacheTargets: array<atomic<u32>, CACHE_SLOTS>;
var<workgroup> cacheLow:     array<atomic<u32>, CACHE_SLOTS>;
var<workgroup> cacheHigh:    array<atomic<u32>, CACHE_SLOTS>;

fn addFixed64ToCache(slot: u32, words: vec2u) {
  let previousLow = atomicAdd(&cacheLow[slot], words.x);
  let carry = select(0u, 1u, previousLow + words.x < previousLow);
  let highIncrement = words.y + carry;
  if (highIncrement != 0u) { atomicAdd(&cacheHigh[slot], highIncrement); }
}

fn claimsCacheSlot(slot: u32, targetCell: u32) -> bool {
  let occupant = atomicLoad(&cacheTargets[slot]);
  if (occupant == targetCell) { return true; }
  if (occupant != EMPTY_SLOT) { return false; }
  let claim = atomicCompareExchangeWeak(&cacheTargets[slot], EMPTY_SLOT, targetCell);
  return claim.exchanged || claim.old_value == targetCell;
}

fn submitContribution(contribution: PairContribution, bufferOffset: u32, laneIndex: u32) {
  if (!contribution.valid) { return; }
  let words = toFixed64(contribution.value);
  let slot  = (contribution.targetIndex * FIBONACCI_HASH) >> CACHE_HASH_SHIFT;
  if (claimsCacheSlot(slot, contribution.targetIndex)) {
    addFixed64ToCache(slot, words);
  } else {
    addFixed64ToStorage(bufferOffset + contribution.targetIndex, words);
  }
}

fn resetCache(localIndex: u32) {
  for (var slot = localIndex; slot < CACHE_SLOTS; slot += ${workgroupSize}u) {
    atomicStore(&cacheTargets[slot], EMPTY_SLOT);
    atomicStore(&cacheLow[slot], 0u);
    atomicStore(&cacheHigh[slot], 0u);
  }
}

fn flushCache(localIndex: u32, bufferOffset: u32) {
  for (var slot = localIndex; slot < CACHE_SLOTS; slot += ${workgroupSize}u) {
    let targetCell = atomicLoad(&cacheTargets[slot]);
    if (targetCell == EMPTY_SLOT) { continue; }
    addFixed64ToStorage(bufferOffset + targetCell, vec2u(atomicLoad(&cacheLow[slot]), atomicLoad(&cacheHigh[slot])));
  }
}
`;
}

// Barriers need every invocation of the workgroup, so the cell checks become a flag
// instead of early returns.
export function tileShape(workgroupSize) {
  const width = workgroupSize >= 128 ? 16 : 8;
  return { width, height: workgroupSize / width };
}

function buildWorkgroupCachedEntryPoint(options) {
  const tiled = (options.cellOrder ?? 'rowMajor') === 'tiled8';
  const workgroupSize = options.workgroupSize ?? 64;
  const tile = tileShape(workgroupSize);
  const cellMapping = tiled ? /* wgsl */`
    let tilesPerRow = (params.gridSize + ${tile.width - 1}u) / ${tile.width}u;
    let tileIndex   = workgroupId.x;
    colA = i32((tileIndex % tilesPerRow) * ${tile.width}u + localIndex % ${tile.width}u);
    rowA = i32((tileIndex / tilesPerRow) * ${tile.height}u + localIndex / ${tile.width}u);
    cellActive = colA < i32(params.gridSize) && rowA < i32(params.gridSize);` : /* wgsl */`
    cellActive = globalId.x < totalCells;
    colA = i32(globalId.x % params.gridSize);
    rowA = i32(globalId.x / params.gridSize);`;

  return /* wgsl */`
@compute @workgroup_size(${workgroupSize})
fn computeRotation(
  @builtin(global_invocation_id)   globalId:    vec3<u32>,
  @builtin(workgroup_id)           workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localIndex:  u32,
) {
  let totalCells   = params.gridSize * params.gridSize;
  let bufferOffset = (workgroupId.x % ACCUMULATION_BUFFERS) * totalCells;
  resetCache(localIndex);
  workgroupBarrier();

  var colA: i32;
  var rowA: i32;
  var cellActive: bool;
  if (params.useMask == 1u) {
    let boxSize = params.maskBoxSize;
    colA = i32(params.maskOriginX) + i32(globalId.x % boxSize);
    rowA = i32(params.maskOriginY) + i32(globalId.x / boxSize);
    cellActive = globalId.x < boxSize * boxSize && insideMaskCircle(colA, rowA)
      && params.minRadius <= 2.0 * sqrt(params.maskRadiusSq);
  } else {
    ${cellMapping}
  }

  let indexA = u32(rowA) * params.gridSize + u32(colA);
  let posA   = vec2f(f32(colA), f32(rowA));
  var velA   = vec2f(0.0);
  if (cellActive) { velA = textureLoad(velocityTexture, vec2i(colA, rowA), 0).xy; }
  cellActive = cellActive && length(velA) >= 1e-6
    && colA % i32(params.sampleStride) == 0 && rowA % i32(params.sampleStride) == 0;

  let gridSize    = i32(params.gridSize);
  let chunkLength = (params.offsetCount + ${options.offsetSplit}u - 1u) / ${options.offsetSplit}u;
  let firstOffset = globalId.y * chunkLength;
  let endOffset   = select(firstOffset, min(params.offsetCount, firstOffset + chunkLength), cellActive);

${buildPairLoop(options, 'submitContribution(contribution, bufferOffset, 0u);')}

  workgroupBarrier();
  flushCache(localIndex, bufferOffset);
}
`;
}

// Integer division by a runtime value is a long instruction sequence on every GPU; the
// operands here are within one period of the range, so a conditional add/subtract
// gives the identical result.
function replaceModuloWrap(source) {
  return source
    .replaceAll('colB = (colB + gridSize) % gridSize;', 'colB += select(select(0, -gridSize, colB >= gridSize), gridSize, colB < 0);')
    .replaceAll('rowB = (rowB + gridSize) % gridSize;', 'rowB += select(select(0, -gridSize, rowB >= gridSize), gridSize, rowB < 0);')
    .replace(`  let gridSize = i32(params.gridSize);
  let centerCol = i32(center.x) % gridSize;
  let centerRow = i32(center.y) % gridSize;
  return u32(centerRow) * params.gridSize + u32(centerCol);`, `  let centerCol = u32(center.x);
  let centerRow = u32(center.y);
  return select(centerRow, 0u, centerRow >= params.gridSize) * params.gridSize
       + select(centerCol, 0u, centerCol >= params.gridSize);`);
}

// Reads velocity from a flat storage buffer (filled by copyTextureToBuffer each frame)
// instead of sampling the texture.
function replaceTextureWithStorage(source) {
  return source
    .replace('@group(0) @binding(0) var velocityTexture: texture_2d<f32>;',
             '@group(0) @binding(0) var<storage, read> velocityBuffer: array<vec2f>;')
    .replace(/textureLoad\(velocityTexture, vec2i\((\w+), (\w+)\), 0\)\.xy/g,
             'velocityBuffer[u32($2) * params.gridSize + u32($1)]');
}

export function buildVariantShader(options) {
  let source = buildVariantShaderSource(options);
  if (options.cheapWrap) source = replaceModuloWrap(source);
  if (options.velocitySource === 'storage') source = replaceTextureWithStorage(source);
  return source;
}

function buildVariantShaderSource(options) {
  const usesSubgroups = options.aggregation === 'subgroup';
  const header = usesSubgroups
    ? 'enable subgroups;\ndiagnostic(off, subgroup_uniformity);\n'
    : '';
  const pairMath = {
    reference:     REFERENCE_PAIR_MATH,
    inverseT:      INVERSE_T_PAIR_MATH,
    inverseTExact: INVERSE_T_EXACT_PAIR_MATH,
  }[options.pairMath];
  if (options.aggregation === 'workgroup') {
    return PARAMS_STRUCT + pairMath + FIXED64_ACCUMULATION
      + buildWorkgroupCache(options.cacheSlots ?? 512, options.workgroupSize ?? 64) + buildWorkgroupCachedEntryPoint(options);
  }
  const submission = usesSubgroups
    ? buildSubgroupAggregation(options.aggregationRounds ?? 1)
    : DIRECT_SUBMISSION;
  return header + PARAMS_STRUCT + pairMath + ACCUMULATION_SOURCES[options.accumulation]
    + submission + buildEntryPoint(options);
}

export function buildReduceShader(accumulation) {
  const accumulationType = accumulation === 'fixed64' ? 'u32' : 'i32';
  const readCell = {
    casF32:  'bitcast<f32>(accumulation[base + cellIndex])',
    fixed32: 'f32(accumulation[base + cellIndex]) / FIXED_POINT_SCALE',
    fixed64: '(f32(bitcast<i32>(accumulation[2u * (base + cellIndex) + 1u])) * 4294967296.0'
           + ' + f32(accumulation[2u * (base + cellIndex)])) / FIXED_POINT_SCALE',
    none:    'f32(accumulation[base + cellIndex])',
  }[accumulation];

  return /* wgsl */`
struct ReduceParams { gridSize: u32, bufferCount: u32, _pad0: u32, _pad1: u32 }
@group(0) @binding(0) var<storage, read>       accumulation: array<${accumulationType}>;
@group(0) @binding(1) var<storage, read_write> rotationOutput: array<f32>;
@group(0) @binding(2) var<uniform>             params: ReduceParams;
override FIXED_POINT_SCALE: f32 = 4294967296.0;

@compute @workgroup_size(64)
fn reduceBuffers(@builtin(global_invocation_id) globalId: vec3<u32>) {
  let cellIndex  = globalId.x;
  let totalCells = params.gridSize * params.gridSize;
  if (cellIndex >= totalCells) { return; }
  var sum = 0.0;
  for (var k = 0u; k < params.bufferCount; k++) {
    let base = k * totalCells;
    sum += ${readCell};
  }
  rotationOutput[cellIndex] = sum;
}
`;
}

export function accumulationWordsPerCell(accumulation) {
  return accumulation === 'fixed64' ? 2 : 1;
}

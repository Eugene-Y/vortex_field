// Compute shader: instantaneous rotation center field accumulation.
//
// Each thread owns one cell A and walks the annulus as runs of consecutive dCol sharing a
// dRow (half-plane only, so every unordered pair is visited once). For every pair the
// intersection of the two velocity normals receives the pair's mean angular velocity.
//
// Sums are 64-bit fixed point split over two u32 atomics. Field B is built to light up
// vortex cores, so many pairs hit the same cell: a CAS float loop retries serially there,
// a native atomicAdd does not, and integer sums are exact and independent of order.
// A per-workgroup cache of target cells absorbs those hot spots before they reach storage.
//
// When useMask=1, threads cover only the bounding box of the focus circle and the span
// table is split over dispatch y, since a small box alone cannot occupy the GPU.

struct Params {
  gridSize:          u32,   // 0
  accumulationScale: f32,   // 4
  parallelThreshold: f32,   // 8
  spanCount:         u32,   // 12  entries in annulusSpans
  minRadius:         f32,   // 16  annulus inner radius (grid cells) — for mask early-exit
  _pad0:             u32,   // 20
  sampleStride:      u32,   // 24
  maskOriginX:       u32,   // 28
  maskOriginY:       u32,   // 32
  maskBoxSize:       u32,   // 36
  maskCenterX:       f32,   // 40
  maskCenterY:       f32,   // 44
  maskRadiusSq:      f32,   // 48
  useMask:           u32,   // 52
  boundaryMode:      u32,   // 56  0=wrap 1=absorb 2=reflect
  showMask:          u32,   // 60  bit0=show CCW(positive) bit1=show CW(negative); default=3
}                            // 64 bytes

struct PairContribution {
  valid:       bool,
  targetIndex: u32,
  value:       f32,
}

struct CellA {
  col:    i32,
  row:    i32,
  inGrid: bool,
}

@group(0) @binding(0) var velocityTexture: texture_2d<f32>;
// Interleaved [low word, high word] per cell of a two's-complement 64-bit fixed-point sum.
@group(0) @binding(1) var<storage, read_write> accumulation: array<atomic<u32>>;
@group(0) @binding(2) var<uniform> params: Params;
// [dRow, dColStart, dColEnd, unused] — precomputed half-plane annulus runs, dColEnd inclusive.
@group(0) @binding(3) var<storage, read> annulusSpans: array<vec4<i32>>;

override FIXED_POINT_SCALE: f32;

const WORKGROUP_SIZE = 64u;
const TILE_WIDTH     = 8u;
const BOUNDARY_WRAP  = 0u;

const MIN_SPEED_SQUARED      = 1e-12;
const MIN_ARM_LENGTH_SQUARED = 0.01;

const TWO_POW_31    = 2147483648.0;
const TWO_POW_32    = 4294967296.0;
const FIXED64_LIMIT = 4.6e18;

const CACHE_SLOTS      = 256u;
const CACHE_HASH_SHIFT = 24u;  // 32 − log2(CACHE_SLOTS): keeps the top bits of the hash
const FIBONACCI_HASH   = 2654435761u;
const EMPTY_SLOT       = 0xFFFFFFFFu;

const NO_CONTRIBUTION = PairContribution(false, 0u, 0.0);

var<workgroup> cacheTargets: array<atomic<u32>, CACHE_SLOTS>;
var<workgroup> cacheLow:     array<atomic<u32>, CACHE_SLOTS>;
var<workgroup> cacheHigh:    array<atomic<u32>, CACHE_SLOTS>;

fn crossProduct(a: vec2f, b: vec2f) -> f32 {
  return a.x * b.y - a.y * b.x;
}

// Integer division is slow on every GPU; the operand is within one period of the range.
fn wrapIntoGrid(index: i32, gridSize: i32) -> i32 {
  return index + select(select(0, -gridSize, index >= gridSize), gridSize, index < 0);
}

fn nearestTorusImage(vector: vec2f, gridSize: f32) -> vec2f {
  return vector - gridSize * floor((vector + 0.5 * gridSize) / gridSize);
}

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

// Folding a center into [0, gridSize) can round up to exactly gridSize; that is cell 0.
fn targetIndexOfCenter(center: vec2f) -> u32 {
  let centerCol = u32(center.x);
  let centerRow = u32(center.y);
  return select(centerRow, 0u, centerRow >= params.gridSize) * params.gridSize
       + select(centerCol, 0u, centerCol >= params.gridSize);
}

fn angularVelocityAboutArm(arm: vec2f, velocity: vec2f) -> f32 {
  return crossProduct(arm, velocity) / dot(arm, arm);
}

// The center lies on both normals: c = posA + t·perp(velA) = posB + s·perp(velB).
// By Cramer's rule every cross product with a perpendicular collapses to a dot:
//   t = dot(δ, velB) / cross(velA, velB),   s = dot(δ, velA) / cross(velA, velB),
// and the arms from c to the cells are −t·perp(velA) and −s·perp(velB).
// In Wrap mode ω is measured from the folded center to the nearest torus image of each
// cell, so the arms are folded as well.
fn evaluatePair(posA: vec2f, velA: vec2f, offset: vec2i, velB: vec2f) -> PairContribution {
  let speedSquaredA = dot(velA, velA);
  let speedSquaredB = dot(velB, velB);
  if (speedSquaredB < MIN_SPEED_SQUARED) { return NO_CONTRIBUTION; }

  let velocityCross = crossProduct(velA, velB);
  let threshold     = params.parallelThreshold;
  if (velocityCross * velocityCross < threshold * threshold * speedSquaredA * speedSquaredB) {
    return NO_CONTRIBUTION;
  }

  let delta      = vec2f(offset);
  let parameterA = dot(delta, velB) / velocityCross;
  let parameterB = dot(delta, velA) / velocityCross;

  let gridSize = f32(params.gridSize);
  var center   = posA + parameterA * vec2f(-velA.y, velA.x);
  var armA     = parameterA * vec2f(velA.y, -velA.x);
  var armB     = parameterB * vec2f(velB.y, -velB.x);
  if (params.boundaryMode == BOUNDARY_WRAP) {
    center -= gridSize * floor(center / gridSize);
    armA    = nearestTorusImage(armA, gridSize);
    armB    = nearestTorusImage(armB, gridSize);
  } else if (center.x < 0.0 || center.x >= gridSize || center.y < 0.0 || center.y >= gridSize) {
    return NO_CONTRIBUTION;
  }

  if (dot(armB, armB) < MIN_ARM_LENGTH_SQUARED) { return NO_CONTRIBUTION; }
  let omegaA = select(angularVelocityAboutArm(armA, velA), 0.0, dot(armA, armA) < MIN_ARM_LENGTH_SQUARED);
  let omegaB = angularVelocityAboutArm(armB, velB);
  let value  = (omegaA + omegaB) * 0.5 * params.accumulationScale / (gridSize * gridSize);
  if (!passesSignFilter(value)) { return NO_CONTRIBUTION; }
  return PairContribution(true, targetIndexOfCenter(center), value);
}

// Returns (low word, high word) of round(value · FIXED_POINT_SCALE) in two's complement.
fn toFixed64(value: f32) -> vec2u {
  let scaled = clamp(round(value * FIXED_POINT_SCALE), -FIXED64_LIMIT, FIXED64_LIMIT);
  if (abs(scaled) < TWO_POW_31) {
    let narrow = i32(scaled);
    return vec2u(bitcast<u32>(narrow), bitcast<u32>(narrow >> 31u));
  }
  let highPart = floor(scaled / TWO_POW_32);
  return vec2u(u32(scaled - highPart * TWO_POW_32), bitcast<u32>(i32(highPart)));
}

// The previous low word returned by the atomic tells this invocation whether its own add
// carried; addition commutes, so the high word converges to the exact sum in any order.
// Small values of either sign leave the high increment at zero (−1 + carry), so most adds
// touch one word.
fn addFixed64ToStorage(cellIndex: u32, words: vec2u) {
  let previousLow   = atomicAdd(&accumulation[2u * cellIndex], words.x);
  let carry         = select(0u, 1u, previousLow + words.x < previousLow);
  let highIncrement = words.y + carry;
  if (highIncrement != 0u) { atomicAdd(&accumulation[2u * cellIndex + 1u], highIncrement); }
}

fn addFixed64ToCache(slot: u32, words: vec2u) {
  let previousLow   = atomicAdd(&cacheLow[slot], words.x);
  let carry         = select(0u, 1u, previousLow + words.x < previousLow);
  let highIncrement = words.y + carry;
  if (highIncrement != 0u) { atomicAdd(&cacheHigh[slot], highIncrement); }
}

// Direct-mapped: the first target hashed to a slot owns it for the rest of the workgroup;
// colliding targets go straight to storage.
fn claimsCacheSlot(slot: u32, targetCell: u32) -> bool {
  let occupant = atomicLoad(&cacheTargets[slot]);
  if (occupant == targetCell) { return true; }
  if (occupant != EMPTY_SLOT) { return false; }
  let claim = atomicCompareExchangeWeak(&cacheTargets[slot], EMPTY_SLOT, targetCell);
  return claim.exchanged || claim.old_value == targetCell;
}

fn submitContribution(contribution: PairContribution) {
  if (!contribution.valid) { return; }
  let words = toFixed64(contribution.value);
  let slot  = (contribution.targetIndex * FIBONACCI_HASH) >> CACHE_HASH_SHIFT;
  if (claimsCacheSlot(slot, contribution.targetIndex)) {
    addFixed64ToCache(slot, words);
  } else {
    addFixed64ToStorage(contribution.targetIndex, words);
  }
}

fn resetCache(localIndex: u32) {
  for (var slot = localIndex; slot < CACHE_SLOTS; slot += WORKGROUP_SIZE) {
    atomicStore(&cacheTargets[slot], EMPTY_SLOT);
    atomicStore(&cacheLow[slot], 0u);
    atomicStore(&cacheHigh[slot], 0u);
  }
}

fn flushCache(localIndex: u32) {
  for (var slot = localIndex; slot < CACHE_SLOTS; slot += WORKGROUP_SIZE) {
    let targetCell = atomicLoad(&cacheTargets[slot]);
    if (targetCell == EMPTY_SLOT) { continue; }
    addFixed64ToStorage(targetCell, vec2u(atomicLoad(&cacheLow[slot]), atomicLoad(&cacheHigh[slot])));
  }
}

// Without a mask each workgroup covers an 8×8 tile, so one span step of the workgroup
// lands on a compact patch of B cells and, in smooth flow, of target cells.
fn resolveCellA(globalId: vec3u, workgroupId: vec3u, localIndex: u32) -> CellA {
  if (params.useMask == 1u) {
    let boxSize = params.maskBoxSize;
    let col = i32(params.maskOriginX + globalId.x % boxSize);
    let row = i32(params.maskOriginY + globalId.x / boxSize);
    return CellA(col, row, globalId.x < boxSize * boxSize && insideMaskCircle(col, row));
  }
  let tilesPerRow = (params.gridSize + TILE_WIDTH - 1u) / TILE_WIDTH;
  let col = i32((workgroupId.x % tilesPerRow) * TILE_WIDTH + localIndex % TILE_WIDTH);
  let row = i32((workgroupId.x / tilesPerRow) * TILE_WIDTH + localIndex / TILE_WIDTH);
  return CellA(col, row, col < i32(params.gridSize) && row < i32(params.gridSize));
}

fn isSampledCell(cell: CellA) -> bool {
  let stride = i32(params.sampleStride);
  return cell.col % stride == 0 && cell.row % stride == 0;
}

// No pair inside the mask circle can reach the annulus inner radius.
fn annulusReachableInsideMask() -> bool {
  return params.useMask == 0u || params.minRadius <= 2.0 * sqrt(params.maskRadiusSq);
}

// Outside Wrap mode the span is clipped to the domain once instead of per pair.
fn accumulateSpan(cell: CellA, velA: vec2f, span: vec4<i32>) {
  let gridSize  = i32(params.gridSize);
  let dRow      = span.x;
  var dColStart = span.y;
  var dColEnd   = span.z;
  var rowB      = cell.row + dRow;
  if (params.boundaryMode == BOUNDARY_WRAP) {
    rowB = wrapIntoGrid(rowB, gridSize);
  } else {
    if (rowB < 0 || rowB >= gridSize) { return; }
    dColStart = max(dColStart, -cell.col);
    dColEnd   = min(dColEnd, gridSize - 1 - cell.col);
  }

  let posA = vec2f(f32(cell.col), f32(cell.row));
  var colB = wrapIntoGrid(cell.col + dColStart, gridSize);
  for (var dCol = dColStart; dCol <= dColEnd; dCol++) {
    if (params.useMask == 0u || insideMaskCircle(colB, rowB)) {
      let velB = textureLoad(velocityTexture, vec2i(colB, rowB), 0).xy;
      submitContribution(evaluatePair(posA, velA, vec2i(dCol, dRow), velB));
    }
    colB += 1;
    if (colB == gridSize) { colB = 0; }
  }
}

fn accumulateSpanChunk(cell: CellA, velA: vec2f, chunkIndex: u32, chunkCount: u32) {
  let chunkLength = (params.spanCount + chunkCount - 1u) / chunkCount;
  let firstSpan   = chunkIndex * chunkLength;
  let endSpan     = min(params.spanCount, firstSpan + chunkLength);
  for (var spanIndex = firstSpan; spanIndex < endSpan; spanIndex++) {
    accumulateSpan(cell, velA, annulusSpans[spanIndex]);
  }
}

// Barriers need every invocation of the workgroup, so idle cells skip the work
// instead of returning early.
@compute @workgroup_size(WORKGROUP_SIZE)
fn computeRotation(
  @builtin(global_invocation_id)   globalId:      vec3<u32>,
  @builtin(workgroup_id)           workgroupId:   vec3<u32>,
  @builtin(num_workgroups)         workgroupGrid: vec3<u32>,
  @builtin(local_invocation_index) localIndex:    u32,
) {
  resetCache(localIndex);
  workgroupBarrier();

  let cell = resolveCellA(globalId, workgroupId, localIndex);
  var velA = vec2f(0.0);
  if (cell.inGrid) { velA = textureLoad(velocityTexture, vec2i(cell.col, cell.row), 0).xy; }
  let contributes = cell.inGrid && dot(velA, velA) >= MIN_SPEED_SQUARED
    && isSampledCell(cell) && annulusReachableInsideMask();
  if (contributes) { accumulateSpanChunk(cell, velA, workgroupId.y, workgroupGrid.y); }

  workgroupBarrier();
  flushCache(localIndex);
}

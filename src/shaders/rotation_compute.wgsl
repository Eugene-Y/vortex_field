// Compute shader: instantaneous rotation center field accumulation.
//
// Each thread handles one cell i and iterates over all j-neighbors listed in
// annulusOffsets (precomputed CPU-side annulus ring, unique pairs only via indexB > indexA).
//
// When useMask=1, threads cover only the bounding box of a focus circle
// (maskBoxSize² threads instead of gridSize²) and both cells must be
// within the circle. This gives O(R²) dispatch vs O(N²) for a circle
// of radius R, without any dead work beyond the mask-rejection check.
//
// Accumulation uses a CAS-based f32 atomic add (bit-cast through i32).
// K private buffers (selected by workgroup_id % K) reduce CAS contention K-fold.

struct Params {
  gridSize:          u32,   // 0
  accumulationScale: f32,   // 4
  parallelThreshold: f32,   // 8
  offsetCount:       u32,   // 12  entries in annulusOffsets
  minRadius:         f32,   // 16  annulus inner radius (grid cells) — for mask early-exit
  maxRadius:         f32,   // 20  annulus outer radius (grid cells) — for mask diameter clamp
  sampleStride:      u32,   // 24
  maskOriginX:       u32,   // 28
  maskOriginY:       u32,   // 32
  maskBoxSize:       u32,   // 36
  maskCenterX:       f32,   // 40
  maskCenterY:       f32,   // 44
  maskRadiusSq:      f32,   // 48
  useMask:           u32,   // 52
  boundaryMode:      u32,   // 56  0=wrap 1=absorb 2=reflect
  pad1:              u32,   // 60
}                            // 64 bytes

@group(0) @binding(0) var velocityTexture: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> rotationBuffer: array<atomic<i32>>;
@group(0) @binding(2) var<uniform> params: Params;
// Flat i32 array: [dCol0, dRow0, dCol1, dRow1, ...] — precomputed annulus offsets.
@group(0) @binding(3) var<storage, read> annulusOffsets: array<i32>;

// CAS-based f32 atomic add — no quantization error.
// Stores f32 bit-pattern in an atomic<i32>; clearBuffer(0) initialises to 0.0 correctly.
fn atomicAddF32(idx: u32, value: f32) {
  var expected = atomicLoad(&rotationBuffer[idx]);
  loop {
    let desired = bitcast<i32>(bitcast<f32>(expected) + value);
    let result  = atomicCompareExchangeWeak(&rotationBuffer[idx], expected, desired);
    if (result.exchanged) { break; }
    expected = result.old_value;
  }
}

fn periodicDisplacement(posA: vec2f, posB: vec2f, gridSize: f32) -> vec2f {
  var delta = posB - posA;
  delta -= gridSize * floor((delta + 0.5 * gridSize) / gridSize);
  return delta;
}

// Like periodicDisplacement but respects boundaryMode:
// in Wrap mode (0) uses shortest torus path; otherwise direct vector.
fn pairDisplacement(posA: vec2f, posB: vec2f, gridSize: f32) -> vec2f {
  if (params.boundaryMode == 0u) {
    return periodicDisplacement(posA, posB, gridSize);
  }
  return posB - posA;
}

fn computeInstantaneousRotationCenter(
  posA: vec2f, velA: vec2f,
  posB: vec2f, velB: vec2f,
  gridSize: f32,
) -> vec2f {
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

  // In Wrap mode fold center back into [0, gridSize); caller discards out-of-domain otherwise.
  if (params.boundaryMode == 0u) {
    return rawCenter - gridSize * floor(rawCenter / gridSize);
  }
  return rawCenter;
}

fn computeAngularVelocity(velA: vec2f, center: vec2f, posA: vec2f, gridSize: f32) -> f32 {
  let arm             = pairDisplacement(center, posA, gridSize);
  let armLengthSquared = dot(arm, arm);
  if (armLengthSquared < 0.01) { return 0.0; }
  let cross = arm.x * velA.y - arm.y * velA.x;
  return cross / armLengthSquared;
}

fn insideMaskCircle(col: i32, row: i32) -> bool {
  let dc = f32(col) - params.maskCenterX;
  let dr = f32(row) - params.maskCenterY;
  return dc * dc + dr * dr <= params.maskRadiusSq;
}

@compute @workgroup_size(64)
fn computeRotation(
  @builtin(global_invocation_id) globalId:    vec3<u32>,
  @builtin(workgroup_id)         workgroupId: vec3<u32>,
) {
  let totalCells = params.gridSize * params.gridSize;

  var colA: i32;
  var rowA: i32;

  if (params.useMask == 1u) {
    // Threads cover the mask bounding box only: O(R²) instead of O(N²).
    let boxSize = params.maskBoxSize;
    let flatIdx = globalId.x;
    if (flatIdx >= boxSize * boxSize) { return; }
    colA = i32(params.maskOriginX) + i32(flatIdx % boxSize);
    rowA = i32(params.maskOriginY) + i32(flatIdx / boxSize);
    if (!insideMaskCircle(colA, rowA)) { return; }
  } else {
    let indexA = globalId.x;
    if (indexA >= totalCells) { return; }
    colA = i32(indexA % params.gridSize);
    rowA = i32(indexA / params.gridSize);
  }

  let indexA = u32(rowA) * params.gridSize + u32(colA);
  let gSize  = f32(params.gridSize);
  let posA   = vec2f(f32(colA), f32(rowA));
  let velA   = textureLoad(velocityTexture, vec2i(colA, rowA), 0).xy;
  if (length(velA) < 1e-6) { return; }
  if (colA % i32(params.sampleStride) != 0 || rowA % i32(params.sampleStride) != 0) { return; }

  // When mask is active, bail early if the annulus inner radius exceeds the mask diameter —
  // no pair within the mask circle can satisfy the minimum distance constraint.
  if (params.useMask == 1u) {
    let maskDiameter = 2.0 * sqrt(params.maskRadiusSq);
    if (params.minRadius > maskDiameter) { return; }
  }

  let bufferOffset = (workgroupId.x % ACCUMULATION_BUFFERS) * totalCells;

  // Iterate precomputed annulus offsets — no per-pair distance check needed.
  for (var k = 0u; k < params.offsetCount; k++) {
    let dCol = annulusOffsets[k * 2u];
    let dRow = annulusOffsets[k * 2u + 1u];

    var colB: i32;
    var rowB: i32;
    if (params.boundaryMode == 0u) {
      colB = ((colA + dCol) % i32(params.gridSize) + i32(params.gridSize)) % i32(params.gridSize);
      rowB = ((rowA + dRow) % i32(params.gridSize) + i32(params.gridSize)) % i32(params.gridSize);
    } else {
      colB = colA + dCol;
      rowB = rowA + dRow;
      if (colB < 0 || colB >= i32(params.gridSize) ||
          rowB < 0 || rowB >= i32(params.gridSize)) { continue; }
    }
    let indexB = u32(rowB) * params.gridSize + u32(colB);

    if (indexB <= indexA) { continue; }

    // When mask is active, discard pairs where cell B is outside the circle.
    if (params.useMask == 1u && !insideMaskCircle(colB, rowB)) { continue; }

    let posB = vec2f(f32(colB), f32(rowB));
    let velB = textureLoad(velocityTexture, vec2i(colB, rowB), 0).xy;

    let center = computeInstantaneousRotationCenter(posA, velA, posB, velB, gSize);
    if (center.x > 1e37) { continue; }

    // In non-Wrap modes discard centers that fall outside the domain.
    if (params.boundaryMode != 0u &&
        (center.x < 0.0 || center.x >= gSize ||
         center.y < 0.0 || center.y >= gSize)) { continue; }

    let armB = pairDisplacement(center, posB, gSize);
    if (dot(armB, armB) < 0.01) { continue; }

    // Average ω from both cells — removes asymmetry from indexA<indexB selection.
    let omegaA = computeAngularVelocity(velA, center, posA, gSize);
    let omegaB = computeAngularVelocity(velB, center, posB, gSize);
    let contribution = (omegaA + omegaB) * 0.5 * params.accumulationScale / f32(totalCells);

    let centerCol   = i32(center.x) % i32(params.gridSize);
    let centerRow   = i32(center.y) % i32(params.gridSize);
    let targetIndex = u32(centerRow) * params.gridSize + u32(centerCol);

    atomicAddF32(bufferOffset + targetIndex, contribution);
  }
}

override ACCUMULATION_BUFFERS: u32 = 16u;

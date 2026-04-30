// Compute shader: instantaneous rotation center field accumulation.
//
// Each thread handles one cell i and iterates over all cells j within
// pairRange radius where j > i (unique pairs only, no double-counting).
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
  pairRange:         f32,   // 12
  sampleStride:      u32,   // 16
  maskOriginX:       u32,   // 20
  maskOriginY:       u32,   // 24
  maskBoxSize:       u32,   // 28
  maskCenterX:       f32,   // 32
  maskCenterY:       f32,   // 36
  maskRadiusSq:      f32,   // 40
  useMask:           u32,   // 44
}                            // 48 bytes

@group(0) @binding(0) var velocityTexture: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> rotationBuffer: array<atomic<i32>>;
@group(0) @binding(2) var<uniform> params: Params;

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

  let delta = periodicDisplacement(posA, posB, gridSize);
  let t     = (delta.x * normalB.y - delta.y * normalB.x) / denominator;
  let rawCenter = posA + t * normalA;

  return rawCenter - gridSize * floor(rawCenter / gridSize);
}

fn computeAngularVelocity(velA: vec2f, center: vec2f, posA: vec2f, gridSize: f32) -> f32 {
  let arm             = periodicDisplacement(center, posA, gridSize);
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

  // Positive pairRange r: pairs within [0,   r·N/2]         — local-first.
  // Negative pairRange r: pairs within [(1+r)·N/2, N/2]     — distant-first.
  // Both ±1 include all pairs; near 0 → very few pairs; at 0 → none.
  let isNegative  = params.pairRange < 0.0;
  let minRadius   = select(0.0, (1.0 + params.pairRange) * gSize * 0.5, isNegative);
  let maxRadius   = select(params.pairRange * gSize * 0.5, gSize * 0.5, isNegative);
  let iMaxRadius  = i32(ceil(maxRadius));
  let minRadiusSq = minRadius * minRadius;
  let maxRadiusSq = maxRadius * maxRadius;

  let bufferOffset = (workgroupId.x % ACCUMULATION_BUFFERS) * totalCells;

  for (var dRow = -iMaxRadius; dRow <= iMaxRadius; dRow++) {
    let dRowSq = f32(dRow) * f32(dRow);
    if (dRowSq > maxRadiusSq) { continue; }

    let iOuterCol = i32(sqrt(maxRadiusSq - dRowSq));
    let iInnerCol = i32(ceil(sqrt(max(0.0, minRadiusSq - dRowSq))));

    for (var wing = 0; wing < 2; wing++) {
      let dColStart = select(-iOuterCol, max(iInnerCol, 1), wing == 1);
      let dColEnd   = select(-iInnerCol, iOuterCol,         wing == 1);
      for (var dCol = dColStart; dCol <= dColEnd; dCol++) {
        let colB   = ((colA + dCol) % i32(params.gridSize) + i32(params.gridSize)) % i32(params.gridSize);
        let rowB   = ((rowA + dRow) % i32(params.gridSize) + i32(params.gridSize)) % i32(params.gridSize);
        let indexB = u32(rowB) * params.gridSize + u32(colB);

        if (indexB <= indexA) { continue; }

        // When mask is active, discard pairs where cell B is outside the circle.
        if (params.useMask == 1u && !insideMaskCircle(colB, rowB)) { continue; }

        let posB     = vec2f(f32(colB), f32(rowB));
        let pairDisp = periodicDisplacement(posA, posB, gSize);
        let distSq   = dot(pairDisp, pairDisp);
        if (distSq > maxRadiusSq || distSq < minRadiusSq) { continue; }

        let velB = textureLoad(velocityTexture, vec2i(colB, rowB), 0).xy;

        let center = computeInstantaneousRotationCenter(posA, velA, posB, velB, gSize);
        if (center.x > 1e37) { continue; }

        let armB = periodicDisplacement(center, posB, gSize);
        if (dot(armB, armB) < 0.01) { continue; }

        let omega        = computeAngularVelocity(velA, center, posA, gSize);
        let contribution = omega * params.accumulationScale / f32(totalCells);

        let centerCol   = i32(center.x) % i32(params.gridSize);
        let centerRow   = i32(center.y) % i32(params.gridSize);
        let targetIndex = u32(centerRow) * params.gridSize + u32(centerCol);

        atomicAddF32(bufferOffset + targetIndex, contribution);
      }
    }
  }
}

override ACCUMULATION_BUFFERS: u32 = 16u;

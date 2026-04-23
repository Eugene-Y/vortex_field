#version 300 es
precision highp float;

// Each vertex corresponds to one pair (i, j) of grid cells.
// gl_VertexID encodes the pair: vertex = i * GRID_SIZE + j.
// We compute the instantaneous center of rotation for that pair
// and emit a point at that position in clip space.

uniform sampler2D u_velocity;
uniform int u_gridSize;
uniform float u_parallelThreshold;
uniform float u_accumulationScale;
// Normalized interaction range [0, 1]. Pairs whose periodic distance exceeds
// pairRange * gridSize / 2 are discarded (diameter = pairRange * gridSize).
// At 1.0 only pairs within the inscribed circle contribute; corner pairs
// (distance > gridSize/2) are excluded at all range values.
uniform float u_pairRange;

out float v_rotationContribution;

vec2 gridIndexToUv(int index) {
  int col = index % u_gridSize;
  int row = index / u_gridSize;
  return (vec2(col, row) + 0.5) / float(u_gridSize);
}

vec2 gridIndexToPosition(int index) {
  int col = index % u_gridSize;
  int row = index / u_gridSize;
  return vec2(col, row);
}

// Shortest displacement from → to on a periodic grid.
// Wraps each component to (-gridSize/2, gridSize/2] via round().
vec2 periodicDisplacement(vec2 from, vec2 to) {
  float gridSize = float(u_gridSize);
  vec2 delta = to - from;
  delta -= gridSize * round(delta / gridSize);
  return delta;
}

// Returns the instantaneous center of rotation for two cells, respecting the
// periodic (wrap-around) topology of the grid. Returns a sentinel (Inf) if the
// velocity vectors are parallel (no finite rotation center exists).
vec2 computeInstantaneousRotationCenter(
  vec2 positionA, vec2 velocityA,
  vec2 positionB, vec2 velocityB
) {
  vec2 normalA = vec2(-velocityA.y, velocityA.x);
  vec2 normalB = vec2(-velocityB.y, velocityB.x);

  float lenA = length(velocityA);
  float lenB = length(velocityB);
  if (lenA < 1e-6 || lenB < 1e-6) {
    return vec2(1.0 / 0.0);
  }

  float denominator = normalA.x * normalB.y - normalA.y * normalB.x;
  if (abs(denominator) / (lenA * lenB) < u_parallelThreshold) {
    return vec2(1.0 / 0.0);
  }

  // Use the periodic (minimum-image) displacement so pairs adjacent across the
  // wrap-around boundary don't produce a spuriously large arm vector.
  vec2 delta = periodicDisplacement(positionA, positionB);
  float t = (delta.x * normalB.y - delta.y * normalB.x) / denominator;
  vec2 rawCenter = positionA + t * normalA;

  // Wrap to [0, gridSize) so the contribution lands in the correct periodic cell.
  return mod(rawCenter, float(u_gridSize));
}

// ω = (arm × vA) / |arm|²  where arm is the vector from center to positionA.
// Uses the periodic arm so cells near a wrap boundary compute the correct ω.
float computeAngularVelocity(vec2 velocityA, vec2 center, vec2 positionA) {
  vec2 arm = periodicDisplacement(center, positionA);
  float armLengthSquared = dot(arm, arm);
  if (armLengthSquared < 0.01) {
    return 0.0;
  }
  float crossProduct = arm.x * velocityA.y - arm.y * velocityA.x;
  return crossProduct / armLengthSquared;
}

void main() {
  int totalCells = u_gridSize * u_gridSize;
  int indexA = gl_VertexID / totalCells;
  int indexB = gl_VertexID % totalCells;

  if (indexA == indexB) {
    gl_Position = vec4(2.0, 2.0, 0.0, 1.0); // outside clip space → discarded
    gl_PointSize = 0.0;
    v_rotationContribution = 0.0;
    return;
  }

  vec2 uvA = gridIndexToUv(indexA);
  vec2 uvB = gridIndexToUv(indexB);
  vec2 velocityA = texture(u_velocity, uvA).xy;
  vec2 velocityB = texture(u_velocity, uvB).xy;
  vec2 positionA = gridIndexToPosition(indexA);
  vec2 positionB = gridIndexToPosition(indexB);

  // Discard pairs whose periodic distance exceeds the configured range.
  // Diameter = pairRange * gridSize, so radius = pairRange * gridSize / 2.
  float rangeRadius = u_pairRange * float(u_gridSize) * 0.5;
  vec2  pairDisplacement = periodicDisplacement(positionA, positionB);
  if (dot(pairDisplacement, pairDisplacement) > rangeRadius * rangeRadius) {
    gl_Position = vec4(2.0, 2.0, 0.0, 1.0);
    gl_PointSize = 0.0;
    v_rotationContribution = 0.0;
    return;
  }

  vec2 center = computeInstantaneousRotationCenter(
    positionA, velocityA, positionB, velocityB
  );

  // After mod(), center is always in [0, gridSize) — only discard true invalids.
  if (isinf(center.x) || isnan(center.x)) {
    gl_Position = vec4(2.0, 2.0, 0.0, 1.0);
    gl_PointSize = 0.0;
    v_rotationContribution = 0.0;
    return;
  }

  float omega = computeAngularVelocity(velocityA, center, positionA);
  v_rotationContribution = omega * u_accumulationScale / float(totalCells);

  vec2 clipPosition = center / float(u_gridSize) * 2.0 - 1.0;
  gl_Position = vec4(clipPosition, 0.0, 1.0);
  gl_PointSize = 1.0;
}

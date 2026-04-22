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

// Returns the instantaneous center of rotation for two cells with given
// positions and velocities, or a sentinel (NaN) if none exists.
// The center is the point equidistant from both where the velocity field
// is consistent with rigid rotation.
// For 2D: the center lies at the intersection of the two perpendiculars
// to the velocity vectors through each cell's position.
vec2 computeInstantaneousRotationCenter(
  vec2 positionA, vec2 velocityA,
  vec2 positionB, vec2 velocityB
) {
  // Perpendicular to velocityA: direction (-vy, vx)
  vec2 normalA = vec2(-velocityA.y, velocityA.x);
  vec2 normalB = vec2(-velocityB.y, velocityB.x);

  // Solve: positionA + t * normalA = positionB + s * normalB
  // positionA - positionB = s * normalB - t * normalA
  vec2 delta = positionB - positionA;
  float denominator = normalA.x * normalB.y - normalA.y * normalB.x;

  // |sin(θ)| between the two velocity vectors: normalised to [0,1].
  // Near zero means velocities are parallel → no meaningful rotation center.
  float lenA = length(velocityA);
  float lenB = length(velocityB);
  if (lenA < 1e-6 || lenB < 1e-6) {
    return vec2(1.0 / 0.0); // one velocity is zero → discard
  }
  if (abs(denominator) / (lenA * lenB) < u_parallelThreshold) {
    return vec2(1.0 / 0.0); // +Inf as sentinel
  }

  float t = (delta.x * normalB.y - delta.y * normalB.x) / denominator;
  return positionA + t * normalA;
}

float computeSignedRotationMagnitude(vec2 velocityA, vec2 velocityB, vec2 center, vec2 positionA) {
  // Sign: cross product of (positionA - center) with velocityA
  // Positive = counter-clockwise, negative = clockwise.
  vec2 arm = positionA - center;
  return arm.x * velocityA.y - arm.y * velocityA.x;
}

void main() {
  int totalCells = u_gridSize * u_gridSize;
  int indexA = gl_VertexID / totalCells;
  int indexB = gl_VertexID % totalCells;

  // Skip self-pairs.
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

  vec2 center = computeInstantaneousRotationCenter(positionA, velocityA, positionB, velocityB);

  // Discard if center is invalid (parallel velocities) or outside the grid.
  if (isinf(center.x) || isnan(center.x) ||
      center.x < 0.0 || center.x >= float(u_gridSize) ||
      center.y < 0.0 || center.y >= float(u_gridSize)) {
    gl_Position = vec4(2.0, 2.0, 0.0, 1.0);
    gl_PointSize = 0.0;
    v_rotationContribution = 0.0;
    return;
  }

  float rotation = computeSignedRotationMagnitude(velocityA, velocityB, center, positionA);
  v_rotationContribution = rotation * u_accumulationScale;

  // Map center grid position to clip space.
  // No +0.5 offset: a float grid coord c maps to screen coord c, which rounds to pixel floor(c).
  vec2 clipPosition = center / float(u_gridSize) * 2.0 - 1.0;
  gl_Position = vec4(clipPosition, 0.0, 1.0);
  gl_PointSize = 1.0;
}

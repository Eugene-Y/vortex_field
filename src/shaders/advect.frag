#version 300 es
precision highp float;

in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D u_velocity;
uniform float u_deltaTime;
uniform float u_gridSize;
uniform float u_damping;
uniform int u_boundary;

vec2 applyBoundary(vec2 uv) {
  if (u_boundary == 0) return fract(uv);
  if (u_boundary == 2) {
    // Mirror once: reflect back into [0,1] without double-wrapping.
    float x = uv.x < 0.0 ? -uv.x : (uv.x > 1.0 ? 2.0 - uv.x : uv.x);
    float y = uv.y < 0.0 ? -uv.y : (uv.y > 1.0 ? 2.0 - uv.y : uv.y);
    return clamp(vec2(x, y), 0.0, 1.0);
  }
  return clamp(uv, 0.0, 1.0);
}

vec2 sampleVelocity(vec2 uv) {
  // Absorb: backtracked positions outside the domain return zero — no energy smuggled in.
  if (u_boundary == 1 && (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0))
    return vec2(0.0);
  return texture(u_velocity, applyBoundary(uv)).xy;
}

// Catmull-Rom weights for fractional position t in [0, 1].
vec4 catmullRomWeights(float t) {
  float t2 = t * t;
  float t3 = t2 * t;
  return vec4(
    0.5 * (-t3 + 2.0*t2 - t),
    0.5 * ( 3.0*t3 - 5.0*t2 + 2.0),
    0.5 * (-3.0*t3 + 4.0*t2 + t),
    0.5 * ( t3 - t2)
  );
}

// Bicubic Catmull-Rom interpolation over a 4x4 neighbourhood.
// More isotropic than bilinear: treats diagonal directions equally to axis-aligned ones.
vec2 sampleVelocityBicubic(vec2 uv) {
  float h  = 1.0 / u_gridSize;
  vec2  st = uv / h - 0.5;
  vec2  i  = floor(st);
  vec4  wx = catmullRomWeights(fract(st.x));
  vec4  wy = catmullRomWeights(fract(st.y));

  vec2 result = vec2(0.0);
  for (int row = 0; row < 4; row++) {
    vec2 rowSum = vec2(0.0);
    for (int col = 0; col < 4; col++) {
      vec2 p = (i + vec2(float(col) - 1.0, float(row) - 1.0) + 0.5) * h;
      rowSum += wx[col] * sampleVelocity(p);
    }
    result += wy[row] * rowSum;
  }
  return result;
}

vec2 advectVelocity(vec2 uv) {
  vec2 velocity         = sampleVelocity(uv);
  vec2 previousPosition = uv - velocity * u_deltaTime / u_gridSize;
  return sampleVelocityBicubic(previousPosition) * u_damping;
}

void main() {
  // Absorb: boundary cells advect freely — velocity exits without forced zeroing.
  // Zeroing here creates a no-slip wall that generates pressure reflections.
  fragColor = vec4(advectVelocity(v_uv), 0.0, 1.0);
}

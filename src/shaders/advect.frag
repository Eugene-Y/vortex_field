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

vec2 advectVelocity(vec2 uv) {
  vec2 velocity = sampleVelocity(uv);
  vec2 previousPosition = uv - velocity * u_deltaTime / u_gridSize;
  return sampleVelocity(previousPosition) * u_damping;
}

void main() {
  // Absorb: boundary cells advect freely — velocity exits without forced zeroing.
  // Zeroing here creates a no-slip wall that generates pressure reflections.
  fragColor = vec4(advectVelocity(v_uv), 0.0, 1.0);
}

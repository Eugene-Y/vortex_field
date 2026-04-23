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
  return texture(u_velocity, applyBoundary(uv)).xy;
}

vec2 advectVelocity(vec2 uv) {
  vec2 velocity = sampleVelocity(uv);
  vec2 previousPosition = uv - velocity * u_deltaTime / u_gridSize;
  return sampleVelocity(previousPosition) * u_damping;
}

void main() {
  if (u_boundary == 1) {
    float ts = 1.0 / u_gridSize;
    if (v_uv.x < ts || v_uv.x > 1.0 - ts || v_uv.y < ts || v_uv.y > 1.0 - ts) {
      fragColor = vec4(0.0, 0.0, 0.0, 1.0);
      return;
    }
  }
  fragColor = vec4(advectVelocity(v_uv), 0.0, 1.0);
}

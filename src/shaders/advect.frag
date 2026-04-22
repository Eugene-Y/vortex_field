#version 300 es
precision highp float;

in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D u_velocity;
uniform float u_deltaTime;
uniform float u_gridSize;
uniform float u_damping;

vec2 sampleVelocity(vec2 uv) {
  return texture(u_velocity, uv).xy;
}

// Semi-Lagrangian advection: trace particle backwards through the velocity field.
vec2 advectVelocity(vec2 uv) {
  vec2 velocity = sampleVelocity(uv);
  vec2 previousPosition = uv - velocity * u_deltaTime / u_gridSize;
  return sampleVelocity(previousPosition) * u_damping;
}

void main() {
  fragColor = vec4(advectVelocity(v_uv), 0.0, 1.0);
}

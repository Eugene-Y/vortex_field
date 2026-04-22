#version 300 es
precision highp float;

in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D u_velocity;
uniform float u_texelSize;

// Finite-difference divergence of the velocity field.
float computeDivergence(vec2 uv) {
  float left   = texture(u_velocity, uv + vec2(-u_texelSize, 0.0)).x;
  float right  = texture(u_velocity, uv + vec2( u_texelSize, 0.0)).x;
  float bottom = texture(u_velocity, uv + vec2(0.0, -u_texelSize)).y;
  float top    = texture(u_velocity, uv + vec2(0.0,  u_texelSize)).y;

  return 0.5 * (right - left + top - bottom);
}

void main() {
  fragColor = vec4(computeDivergence(v_uv), 0.0, 0.0, 1.0);
}

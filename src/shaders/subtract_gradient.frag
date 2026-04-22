#version 300 es
precision highp float;

in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D u_velocity;
uniform sampler2D u_pressure;
uniform float u_texelSize;

// Subtract the pressure gradient to enforce incompressibility (∇·u = 0).
vec2 subtractPressureGradient(vec2 uv) {
  float left  = texture(u_pressure, uv + vec2(-u_texelSize, 0.0)).r;
  float right = texture(u_pressure, uv + vec2( u_texelSize, 0.0)).r;
  float bottom = texture(u_pressure, uv + vec2(0.0, -u_texelSize)).r;
  float top    = texture(u_pressure, uv + vec2(0.0,  u_texelSize)).r;

  vec2 gradient = vec2(right - left, top - bottom) * 0.5;
  return texture(u_velocity, uv).xy - gradient;
}

void main() {
  fragColor = vec4(subtractPressureGradient(v_uv), 0.0, 1.0);
}

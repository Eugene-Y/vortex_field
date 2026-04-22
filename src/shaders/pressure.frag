#version 300 es
precision highp float;

in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D u_pressure;
uniform sampler2D u_divergence;
uniform float u_texelSize;

// One Jacobi iteration for the Poisson pressure equation.
float jacobiPressureStep(vec2 uv) {
  float left      = texture(u_pressure, uv + vec2(-u_texelSize, 0.0)).r;
  float right     = texture(u_pressure, uv + vec2( u_texelSize, 0.0)).r;
  float bottom    = texture(u_pressure, uv + vec2(0.0, -u_texelSize)).r;
  float top       = texture(u_pressure, uv + vec2(0.0,  u_texelSize)).r;
  float divergence = texture(u_divergence, uv).r;

  return (left + right + bottom + top - divergence) * 0.25;
}

void main() {
  fragColor = vec4(jacobiPressureStep(v_uv), 0.0, 0.0, 1.0);
}

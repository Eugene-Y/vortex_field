#version 300 es
precision highp float;

in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D u_velocity;
uniform float u_texelSize;
uniform int u_boundary;

bool outsideDomain(vec2 uv) {
  return uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0;
}

float sampleU(vec2 uv) {
  if (u_boundary == 0) return texture(u_velocity, fract(uv)).x;
  // Absorb and reflect: no normal flow through/past boundary → 0 outside domain.
  if (outsideDomain(uv)) return 0.0;
  return texture(u_velocity, uv).x;
}

float sampleV(vec2 uv) {
  if (u_boundary == 0) return texture(u_velocity, fract(uv)).y;
  if (outsideDomain(uv)) return 0.0;
  return texture(u_velocity, uv).y;
}

float computeDivergence(vec2 uv) {
  float left   = sampleU(uv + vec2(-u_texelSize, 0.0));
  float right  = sampleU(uv + vec2( u_texelSize, 0.0));
  float bottom = sampleV(uv + vec2(0.0, -u_texelSize));
  float top    = sampleV(uv + vec2(0.0,  u_texelSize));

  return 0.5 * (right - left + top - bottom);
}

void main() {
  fragColor = vec4(computeDivergence(v_uv), 0.0, 0.0, 1.0);
}

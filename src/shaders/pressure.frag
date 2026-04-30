#version 300 es
precision highp float;

in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D u_pressure;
uniform sampler2D u_divergence;
uniform float u_texelSize;
uniform int u_boundary;

float samplePressure(vec2 uv) {
  if (u_boundary == 0) return texture(u_pressure, fract(uv)).r;
  // Absorb: p=0 (Dirichlet) — open boundary, consistent with subtract_gradient.
  // Reflect: Neumann (clamp) — solid wall, zero pressure gradient.
  if (u_boundary == 1 &&
      (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0)) return 0.0;
  return texture(u_pressure, clamp(uv, 0.0, 1.0)).r;
}

float jacobiPressureStep(vec2 uv) {
  float left   = samplePressure(uv + vec2(-u_texelSize,  0.0));
  float right  = samplePressure(uv + vec2( u_texelSize,  0.0));
  float bottom = samplePressure(uv + vec2( 0.0, -u_texelSize));
  float top    = samplePressure(uv + vec2( 0.0,  u_texelSize));
  float sw     = samplePressure(uv + vec2(-u_texelSize, -u_texelSize));
  float se     = samplePressure(uv + vec2( u_texelSize, -u_texelSize));
  float nw     = samplePressure(uv + vec2(-u_texelSize,  u_texelSize));
  float ne     = samplePressure(uv + vec2( u_texelSize,  u_texelSize));
  float divergence = texture(u_divergence, uv).r;

  // Mehrstellen 9-point isotropic Laplacian: axis-aligned ×4, diagonals ×1.
  // Distributes pressure equally in all 8 directions vs axis-only 5-point stencil.
  return (4.0 * (left + right + bottom + top) + (sw + se + nw + ne) - 6.0 * divergence) / 20.0;
}

void main() {
  fragColor = vec4(jacobiPressureStep(v_uv), 0.0, 0.0, 1.0);
}

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
  // Absorb and reflect both use Neumann (zero gradient) — clamp to edge.
  // Dirichlet (p=0) for absorb would create an inward pressure gradient, causing reflections.
  return texture(u_pressure, clamp(uv, 0.0, 1.0)).r;
}

float jacobiPressureStep(vec2 uv) {
  float left      = samplePressure(uv + vec2(-u_texelSize, 0.0));
  float right     = samplePressure(uv + vec2( u_texelSize, 0.0));
  float bottom    = samplePressure(uv + vec2(0.0, -u_texelSize));
  float top       = samplePressure(uv + vec2(0.0,  u_texelSize));
  float divergence = texture(u_divergence, uv).r;

  return (left + right + bottom + top - divergence) * 0.25;
}

void main() {
  fragColor = vec4(jacobiPressureStep(v_uv), 0.0, 0.0, 1.0);
}

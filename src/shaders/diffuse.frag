#version 300 es
precision highp float;

in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D u_velocity;
uniform sampler2D u_velocityPrev;
uniform float u_alpha;
uniform float u_beta;
uniform float u_texelSize;
uniform int u_boundary;

vec2 sampleVelocity(vec2 uv) {
  if (u_boundary == 0) return texture(u_velocity, fract(uv)).xy;
  // Both absorb and reflect use Neumann (clamp) for diffusion — velocity is continuous
  // across the boundary so fluid exits smoothly without a forced zero wall.
  return texture(u_velocity, clamp(uv, 0.0, 1.0)).xy;
}

vec2 jacobiDiffuseStep(vec2 uv) {
  vec2 left   = sampleVelocity(uv + vec2(-u_texelSize, 0.0));
  vec2 right  = sampleVelocity(uv + vec2( u_texelSize, 0.0));
  vec2 bottom = sampleVelocity(uv + vec2(0.0, -u_texelSize));
  vec2 top    = sampleVelocity(uv + vec2(0.0,  u_texelSize));
  vec2 center = texture(u_velocityPrev, uv).xy;

  return (left + right + bottom + top + u_alpha * center) * u_beta;
}

void main() {
  fragColor = vec4(jacobiDiffuseStep(v_uv), 0.0, 1.0);
}

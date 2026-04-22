#version 300 es
precision highp float;

in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D u_velocity;
uniform sampler2D u_velocityPrev;
uniform float u_alpha;   // (cellSize^2) / (viscosity * deltaTime)
uniform float u_beta;    // 1.0 / (4.0 + alpha)
uniform float u_texelSize; // 1.0 / gridSize

// One Jacobi iteration for viscous diffusion.
vec2 jacobiDiffuseStep(vec2 uv) {
  vec2 left   = texture(u_velocity, uv + vec2(-u_texelSize, 0.0)).xy;
  vec2 right  = texture(u_velocity, uv + vec2( u_texelSize, 0.0)).xy;
  vec2 bottom = texture(u_velocity, uv + vec2(0.0, -u_texelSize)).xy;
  vec2 top    = texture(u_velocity, uv + vec2(0.0,  u_texelSize)).xy;
  vec2 center = texture(u_velocityPrev, uv).xy;

  return (left + right + bottom + top + u_alpha * center) * u_beta;
}

void main() {
  fragColor = vec4(jacobiDiffuseStep(v_uv), 0.0, 1.0);
}

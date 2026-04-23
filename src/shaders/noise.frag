#version 300 es
precision highp float;

in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D u_velocity;
uniform float u_seed;
uniform float u_strength;

float pseudoRandom(vec2 uv, float seed) {
  return fract(sin(dot(uv + seed, vec2(127.1, 311.7))) * 43758.5453);
}

void main() {
  vec2 existing = texture(u_velocity, v_uv).xy;
  float angle = pseudoRandom(v_uv, u_seed) * 6.28318;
  vec2 noise = vec2(cos(angle), sin(angle)) * u_strength;
  fragColor = vec4(existing + noise, 0.0, 1.0);
}

#version 300 es
precision highp float;

in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D u_velocity;
uniform vec2 u_impulsePosition; // normalized [0,1]
uniform vec2 u_impulseDirection;
uniform float u_impulseRadius;  // in UV space
uniform float u_impulseStrength;

vec2 gaussianImpulse(vec2 uv) {
  vec2 offset = uv - u_impulsePosition;
  float distanceSquared = dot(offset, offset);
  float radiusSquared = u_impulseRadius * u_impulseRadius;
  float weight = exp(-distanceSquared / radiusSquared);
  return u_impulseDirection * u_impulseStrength * weight;
}

void main() {
  vec2 existing = texture(u_velocity, v_uv).xy;
  fragColor = vec4(existing + gaussianImpulse(v_uv), 0.0, 1.0);
}

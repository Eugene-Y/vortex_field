#version 300 es
precision highp float;

in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D u_velocity;
uniform vec2 u_impulsePosition;
uniform vec2 u_impulseDirection;
uniform float u_impulseRadius;
uniform float u_impulseStrength;
uniform int u_boundary;
uniform float u_gridSize;

vec2 gaussianImpulse(vec2 uv) {
  vec2 offset = uv - u_impulsePosition;
  if (u_boundary == 0) offset -= round(offset); // periodic minimum-image
  float distanceSquared = dot(offset, offset);
  float radiusSquared = u_impulseRadius * u_impulseRadius;
  float weight = exp(-distanceSquared / radiusSquared);
  return u_impulseDirection * u_impulseStrength * weight;
}

void main() {
  if (u_boundary == 1) {
    float ts = 1.0 / u_gridSize;
    if (v_uv.x < ts || v_uv.x > 1.0 - ts || v_uv.y < ts || v_uv.y > 1.0 - ts) {
      fragColor = vec4(texture(u_velocity, v_uv).xy, 0.0, 1.0);
      return;
    }
  }
  vec2 existing = texture(u_velocity, v_uv).xy;
  fragColor = vec4(existing + gaussianImpulse(v_uv), 0.0, 1.0);
}

#version 300 es
precision highp float;

in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D u_velocity;
uniform vec2  u_center;
uniform float u_radius;
uniform float u_strength;
uniform int   u_mode;   // 0 = spin (CCW), 1 = explode, 2 = implode
uniform int   u_boundary;
uniform float u_gridSize;

// Returns the velocity direction for this fragment's position within the disk.
vec2 diskVelocity(vec2 offset) {
  float dist = length(offset);
  if (dist < 1e-6) return vec2(0.0);

  vec2 radial  =  offset / dist;
  vec2 tangent = vec2(-radial.y, radial.x); // CCW perpendicular

  if (u_mode == 0) return tangent;
  if (u_mode == 1) return radial;
  return -radial;
}

void main() {
  if (u_boundary == 1) {
    float ts = 1.0 / u_gridSize;
    if (v_uv.x < ts || v_uv.x > 1.0 - ts || v_uv.y < ts || v_uv.y > 1.0 - ts) {
      fragColor = vec4(texture(u_velocity, v_uv).xy, 0.0, 1.0);
      return;
    }
  }

  vec2  offset = v_uv - u_center;
  float dist   = length(offset);

  vec2 existing = texture(u_velocity, v_uv).xy;

  if (dist > u_radius) {
    fragColor = vec4(existing, 0.0, 1.0);
    return;
  }

  // Gaussian radial falloff eliminates the sharp velocity boundary that would
  // otherwise create a divergence ring and attract the pressure correction there.
  float t    = dist / u_radius;
  float edge = exp(-3.5 * t * t);
  vec2  added   = diskVelocity(offset) * u_strength * edge;
  fragColor = vec4(existing + added, 0.0, 1.0);
}

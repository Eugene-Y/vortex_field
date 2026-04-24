#version 300 es
precision highp float;

in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D u_velocity;
uniform float u_texelSize;

// Raw finite-difference curl (z-component of ∇×v) in texture-space units.
// Not divided by texelSize — the confinement shader cancels it out anyway.
float computeCurl(vec2 uv) {
  float dvydx = texture(u_velocity, uv + vec2( u_texelSize, 0.0)).y
              - texture(u_velocity, uv + vec2(-u_texelSize, 0.0)).y;
  float dvxdy = texture(u_velocity, uv + vec2(0.0,  u_texelSize)).x
              - texture(u_velocity, uv + vec2(0.0, -u_texelSize)).x;
  return (dvydx - dvxdy) * 0.5;
}

void main() {
  fragColor = vec4(computeCurl(v_uv), 0.0, 0.0, 1.0);
}

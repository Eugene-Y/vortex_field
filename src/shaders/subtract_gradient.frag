#version 300 es
precision highp float;

in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D u_velocity;
uniform sampler2D u_pressure;
uniform float u_texelSize;
uniform int u_boundary;

float samplePressure(vec2 uv) {
  if (u_boundary == 0) return texture(u_pressure, fract(uv)).r;
  // Absorb: p=0 at ghost cells (Dirichlet) — open boundary, fluid exits freely.
  // Reflect: Neumann (clamp, zero gradient) — solid wall.
  if (u_boundary == 1 &&
      (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0)) return 0.0;
  return texture(u_pressure, clamp(uv, 0.0, 1.0)).r;
}

vec2 subtractPressureGradient(vec2 uv) {
  float left   = samplePressure(uv + vec2(-u_texelSize, 0.0));
  float right  = samplePressure(uv + vec2( u_texelSize, 0.0));
  float bottom = samplePressure(uv + vec2(0.0, -u_texelSize));
  float top    = samplePressure(uv + vec2(0.0,  u_texelSize));

  vec2 gradient = vec2(right - left, top - bottom) * 0.5;
  return texture(u_velocity, uv).xy - gradient;
}

void main() {
  bool onLeft   = v_uv.x < u_texelSize;
  bool onRight  = v_uv.x > 1.0 - u_texelSize;
  bool onBottom = v_uv.y < u_texelSize;
  bool onTop    = v_uv.y > 1.0 - u_texelSize;
  bool onBoundary = onLeft || onRight || onBottom || onTop;

  vec2 vel = subtractPressureGradient(v_uv);

  if (u_boundary == 2 && onBoundary) {
    // Reflect: negate the normal component at each wall.
    if (onLeft || onRight)  vel.x = -vel.x;
    if (onBottom || onTop)  vel.y = -vel.y;
  }

  fragColor = vec4(vel, 0.0, 1.0);
}

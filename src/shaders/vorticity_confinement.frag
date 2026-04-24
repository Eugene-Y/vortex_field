#version 300 es
precision highp float;

in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D u_velocity;
uniform sampler2D u_curl;
uniform float u_texelSize;
uniform float u_strength;
uniform float u_deltaTime;

// Gradient of |curl| magnitude in raw (non-normalised) texture units.
vec2 curlMagnitudeGradient(vec2 uv) {
  float left   = abs(texture(u_curl, uv + vec2(-u_texelSize, 0.0)).r);
  float right  = abs(texture(u_curl, uv + vec2( u_texelSize, 0.0)).r);
  float bottom = abs(texture(u_curl, uv + vec2(0.0, -u_texelSize)).r);
  float top    = abs(texture(u_curl, uv + vec2(0.0,  u_texelSize)).r);
  return vec2(right - left, top - bottom) * 0.5;
}

// Re-energises vortices that numerical diffusion would otherwise smooth away.
// Pushes fluid tangentially around vortex cores proportional to local curl.
vec2 computeConfinementForce(vec2 uv) {
  float curl = texture(u_curl, uv).r;
  vec2  grad = curlMagnitudeGradient(uv);
  float len  = length(grad) + 1e-5;
  vec2  eta  = grad / len;                     // unit vector toward stronger curl
  return u_strength * curl * vec2(eta.y, -eta.x);
}

void main() {
  vec2 velocity = texture(u_velocity, v_uv).xy;
  vec2 force    = computeConfinementForce(v_uv);
  fragColor = vec4(velocity + force * u_deltaTime, 0.0, 1.0);
}

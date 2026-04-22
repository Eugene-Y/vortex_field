#version 300 es
precision highp float;

in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D u_field;
uniform int u_mode; // 0 = velocity magnitude, 1 = rotation

uniform vec3 u_colorPositive;
uniform vec3 u_colorNegative;
uniform float u_rotationToneMidpoint; // Reinhard midpoint: value that maps to 50% brightness

vec3 hsvToRgb(vec3 hsv) {
  float h = hsv.x * 6.0;
  float s = hsv.y;
  float v = hsv.z;
  float c = v * s;
  float x = c * (1.0 - abs(mod(h, 2.0) - 1.0));
  float m = v - c;
  vec3 rgb;
  if      (h < 1.0) rgb = vec3(c, x, 0.0);
  else if (h < 2.0) rgb = vec3(x, c, 0.0);
  else if (h < 3.0) rgb = vec3(0.0, c, x);
  else if (h < 4.0) rgb = vec3(0.0, x, c);
  else if (h < 5.0) rgb = vec3(x, 0.0, c);
  else              rgb = vec3(c, 0.0, x);
  return rgb + m;
}

vec3 mapVelocityToColor(vec2 velocity) {
  float magnitude = length(velocity);
  float brightness = clamp(magnitude / 200.0, 0.0, 1.0);
  float angle = atan(velocity.y, velocity.x); // -π to π
  float hue = (angle + 3.14159265) / (2.0 * 3.14159265); // normalize to [0,1]
  return hsvToRgb(vec3(hue, 0.8, brightness));
}

vec3 mapRotationToColor(float rotation) {
  // Reinhard tonemapping: x / (x + c). Never saturates, maps [0,∞) → [0,1) smoothly.
  // At |rotation| = c: brightness = 0.5.
  float absolute = abs(rotation);
  float magnitude = absolute / (absolute + u_rotationToneMidpoint);
  if (rotation > 0.0) {
    return u_colorPositive * magnitude;
  } else {
    return u_colorNegative * magnitude;
  }
}

void main() {
  if (u_mode == 0) {
    vec2 velocity = texture(u_field, v_uv).xy;
    fragColor = vec4(mapVelocityToColor(velocity), 1.0);
  } else {
    float rotation = texture(u_field, v_uv).r;
    fragColor = vec4(mapRotationToColor(rotation), 1.0);
  }
}

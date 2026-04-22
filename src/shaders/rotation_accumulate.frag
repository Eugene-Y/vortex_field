#version 300 es
precision highp float;

in float v_rotationContribution;
out vec4 fragColor;

void main() {
  // Each fragment adds its rotation contribution to the accumulation buffer.
  // Additive blending (ONE, ONE) is enabled by the caller.
  fragColor = vec4(v_rotationContribution, 0.0, 0.0, 1.0);
}

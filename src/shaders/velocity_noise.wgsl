// Adds per-cell pseudo-random velocity. Used by the noise pattern injector.

struct NoiseParams {
  gridSize: u32,
  pad0:     u32,
  seed:     f32,
  strength: f32,
}

@group(0) @binding(0) var velocityIn:  texture_2d<f32>;
@group(0) @binding(1) var velocityOut: texture_storage_2d<rg32float, write>;
@group(0) @binding(2) var<uniform> params: NoiseParams;

fn pseudoRandom(uv: vec2f, seed: f32) -> f32 {
  return fract(sin(dot(uv + seed, vec2f(127.1, 311.7))) * 43758.5453);
}

@compute @workgroup_size(8, 8)
fn injectNoise(@builtin(global_invocation_id) id: vec3u) {
  let col = i32(id.x);
  let row = i32(id.y);
  let n   = i32(params.gridSize);
  if (col >= n || row >= n) { return; }

  let uv       = vec2f(f32(col), f32(row)) / f32(n);
  let angle    = pseudoRandom(uv, params.seed) * 6.28318;
  let noise    = vec2f(cos(angle), sin(angle)) * params.strength;
  let existing = textureLoad(velocityIn, vec2i(col, row), 0).xy;
  textureStore(velocityOut, vec2i(col, row), vec4f(existing + noise, 0.0, 0.0));
}

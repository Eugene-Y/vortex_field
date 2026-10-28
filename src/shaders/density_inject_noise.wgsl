// Adds per-cell pseudo-random density. Used by the noise pattern in mass mode.

struct DensityNoiseParams {
  gridSize:  u32,
  pad0:      u32,
  seed:      f32,
  strength:  f32,
}

@group(0) @binding(0) var densityIn:  texture_2d<f32>;
@group(0) @binding(1) var densityOut: texture_storage_2d<r32float, write>;
@group(0) @binding(2) var<uniform> params: DensityNoiseParams;

fn pseudoRandom(uv: vec2f, seed: f32) -> f32 {
  return fract(sin(dot(uv + seed, vec2f(127.1, 311.7))) * 43758.5453);
}

@compute @workgroup_size(8, 8)
fn injectDensityNoise(@builtin(global_invocation_id) id: vec3u) {
  let col = i32(id.x);
  let row = i32(id.y);
  let n   = i32(params.gridSize);
  if (col >= n || row >= n) { return; }

  let uv       = vec2f(f32(col), f32(row)) / f32(n);
  let rnd      = pseudoRandom(uv, params.seed);
  let existing = textureLoad(densityIn, vec2i(col, row), 0).r;
  textureStore(densityOut, vec2i(col, row), vec4f(max(0.0, existing + rnd * params.strength), 0.0, 0.0, 0.0));
}

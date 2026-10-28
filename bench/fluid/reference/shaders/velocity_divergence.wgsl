// Computes the velocity divergence ∇·v = ∂vx/∂x + ∂vy/∂y via central differences.

struct BaseParams {
  gridSize:     u32,
  boundaryMode: u32,
  pad0:         u32,
  pad1:         u32,
}

@group(0) @binding(0) var velocityIn:    texture_2d<f32>;
@group(0) @binding(1) var divergenceOut: texture_storage_2d<r32float, write>;
@group(0) @binding(2) var<uniform> params: BaseParams;

fn wrapI(c: i32, n: i32) -> i32 {
  return ((c % n) + n) % n;
}

// Sample vx; outside the domain in non-wrap modes: no normal flow → 0.
fn sampleU(col: i32, row: i32) -> f32 {
  let n = i32(params.gridSize);
  if (params.boundaryMode == 0u) {
    return textureLoad(velocityIn, vec2i(wrapI(col, n), wrapI(row, n)), 0).x;
  }
  if (col < 0 || col >= n || row < 0 || row >= n) { return 0.0; }
  return textureLoad(velocityIn, vec2i(col, row), 0).x;
}

// Sample vy; same boundary treatment as sampleU.
fn sampleV(col: i32, row: i32) -> f32 {
  let n = i32(params.gridSize);
  if (params.boundaryMode == 0u) {
    return textureLoad(velocityIn, vec2i(wrapI(col, n), wrapI(row, n)), 0).y;
  }
  if (col < 0 || col >= n || row < 0 || row >= n) { return 0.0; }
  return textureLoad(velocityIn, vec2i(col, row), 0).y;
}

@compute @workgroup_size(8, 8)
fn computeDivergence(@builtin(global_invocation_id) id: vec3u) {
  let col = i32(id.x);
  let row = i32(id.y);
  let n   = i32(params.gridSize);
  if (col >= n || row >= n) { return; }

  let div = 0.5 * ((sampleU(col + 1, row) - sampleU(col - 1, row))
                 + (sampleV(col, row + 1) - sampleV(col, row - 1)));
  textureStore(divergenceOut, vec2i(col, row), vec4f(div, 0.0, 0.0, 0.0));
}

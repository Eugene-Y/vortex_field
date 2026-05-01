// Subtracts the pressure gradient from velocity to enforce incompressibility.
// Also applies reflect boundary conditions (negate normal component at walls).

struct BaseParams {
  gridSize:     u32,
  boundaryMode: u32,
  pad0:         u32,
  pad1:         u32,
}

@group(0) @binding(0) var velocityIn:  texture_2d<f32>;
@group(0) @binding(1) var pressureIn:  texture_2d<f32>;
@group(0) @binding(2) var velocityOut: texture_storage_2d<rg32float, write>;
@group(0) @binding(3) var<uniform> params: BaseParams;

fn wrapI(c: i32, n: i32) -> i32 {
  return ((c % n) + n) % n;
}

fn samplePressure(col: i32, row: i32) -> f32 {
  let n = i32(params.gridSize);
  if (params.boundaryMode == 0u) {
    return textureLoad(pressureIn, vec2i(wrapI(col, n), wrapI(row, n)), 0).r;
  }
  // Absorb: Dirichlet p=0 — consistent with pressure solve ghost-cell treatment.
  if (params.boundaryMode == 1u && (col < 0 || col >= n || row < 0 || row >= n)) {
    return 0.0;
  }
  return textureLoad(pressureIn, vec2i(clamp(col, 0, n - 1), clamp(row, 0, n - 1)), 0).r;
}

@compute @workgroup_size(8, 8)
fn subtractPressureGradient(@builtin(global_invocation_id) id: vec3u) {
  let col = i32(id.x);
  let row = i32(id.y);
  let n   = i32(params.gridSize);
  if (col >= n || row >= n) { return; }

  let gradient = vec2f(
    samplePressure(col + 1, row) - samplePressure(col - 1, row),
    samplePressure(col, row + 1) - samplePressure(col, row - 1),
  ) * 0.5;

  var velocity = textureLoad(velocityIn, vec2i(col, row), 0).xy - gradient;

  // Reflect: negate the normal velocity component at each boundary cell.
  if (params.boundaryMode == 2u) {
    if (col == 0 || col == n - 1) { velocity.x = -velocity.x; }
    if (row == 0 || row == n - 1) { velocity.y = -velocity.y; }
  }

  textureStore(velocityOut, vec2i(col, row), vec4f(velocity, 0.0, 0.0));
}

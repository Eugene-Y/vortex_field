// Computes the scalar curl (z-component of ∇×v): ω = ∂vy/∂x − ∂vx/∂y.
// Raw finite differences; not divided by cell size — confinement cancels it out.

struct BaseParams {
  gridSize:     u32,
  boundaryMode: u32,
  pad0:         u32,
  pad1:         u32,
}

@group(0) @binding(0) var velocityIn: texture_2d<f32>;
@group(0) @binding(1) var curlOut:    texture_storage_2d<r32float, write>;
@group(0) @binding(2) var<uniform> params: BaseParams;

fn wrapI(c: i32, n: i32) -> i32 {
  return ((c % n) + n) % n;
}

fn sampleVelocity(col: i32, row: i32) -> vec2f {
  let n = i32(params.gridSize);
  if (params.boundaryMode == 0u) {
    return textureLoad(velocityIn, vec2i(wrapI(col, n), wrapI(row, n)), 0).xy;
  }
  return textureLoad(velocityIn, vec2i(clamp(col, 0, n - 1), clamp(row, 0, n - 1)), 0).xy;
}

@compute @workgroup_size(8, 8)
fn computeCurl(@builtin(global_invocation_id) id: vec3u) {
  let col = i32(id.x);
  let row = i32(id.y);
  let n   = i32(params.gridSize);
  if (col >= n || row >= n) { return; }

  let dvydx = sampleVelocity(col + 1, row).y - sampleVelocity(col - 1, row).y;
  let dvxdy = sampleVelocity(col, row + 1).x - sampleVelocity(col, row - 1).x;
  let curl  = (dvydx - dvxdy) * 0.5;
  textureStore(curlOut, vec2i(col, row), vec4f(curl, 0.0, 0.0, 0.0));
}

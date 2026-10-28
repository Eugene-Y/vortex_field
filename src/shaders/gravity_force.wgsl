// Applies gravitational body force to the velocity field.
// Computes g = -∇φ via central differences and adds strength * g * dt to velocity.
// F = -∇φ is attractive: fluid accelerates toward mass concentrations.

struct GravityForceParams {
  gridSize:     u32,
  boundaryMode: u32,
  strength:     f32,
  deltaTime:    f32,
}

@group(0) @binding(0) var velocityIn:  texture_2d<f32>;
@group(0) @binding(1) var potentialIn: texture_2d<f32>;
@group(0) @binding(2) var velocityOut: texture_storage_2d<rg32float, write>;
@group(0) @binding(3) var<uniform> params: GravityForceParams;

fn wrapI(c: i32, n: i32) -> i32 {
  return ((c % n) + n) % n;
}

fn samplePotential(col: i32, row: i32) -> f32 {
  let n = i32(params.gridSize);
  if (params.boundaryMode == 0u) {
    return textureLoad(potentialIn, vec2i(wrapI(col, n), wrapI(row, n)), 0).r;
  }
  return textureLoad(potentialIn, vec2i(clamp(col, 0, n - 1), clamp(row, 0, n - 1)), 0).r;
}

@compute @workgroup_size(8, 8)
fn applyGravityForce(@builtin(global_invocation_id) id: vec3u) {
  let col = i32(id.x);
  let row = i32(id.y);
  let n   = i32(params.gridSize);
  if (col >= n || row >= n) { return; }

  let velocity = textureLoad(velocityIn, vec2i(col, row), 0).xy;

  // Central difference gradient of gravitational potential φ.
  let dPhiDx = (samplePotential(col + 1, row) - samplePotential(col - 1, row)) * 0.5;
  let dPhiDy = (samplePotential(col, row + 1) - samplePotential(col, row - 1)) * 0.5;

  // Gravitational acceleration g = -∇φ, applied as an impulse over dt.
  let accel       = -vec2f(dPhiDx, dPhiDy) * params.strength * params.deltaTime;
  let newVelocity = velocity + accel;

  textureStore(velocityOut, vec2i(col, row), vec4f(newVelocity, 0.0, 0.0));
}

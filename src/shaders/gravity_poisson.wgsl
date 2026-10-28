// One Jacobi iteration of the 9-point Mehrstellen gravity potential solve.
// Solves ∇²φ = ρ — same stencil as the pressure solver, source term = density.
// Warm-started from the previous frame for faster convergence.

struct GravityPoissonParams {
  gridSize:     u32,
  boundaryMode: u32,
  pad0:         u32,
  pad1:         u32,
}

@group(0) @binding(0) var potentialIn:  texture_2d<f32>;
@group(0) @binding(1) var densityIn:    texture_2d<f32>;
@group(0) @binding(2) var potentialOut: texture_storage_2d<r32float, write>;
@group(0) @binding(3) var<uniform> params: GravityPoissonParams;

fn wrapI(c: i32, n: i32) -> i32 {
  return ((c % n) + n) % n;
}

fn samplePotential(col: i32, row: i32) -> f32 {
  let n = i32(params.gridSize);
  if (params.boundaryMode == 0u) {
    return textureLoad(potentialIn, vec2i(wrapI(col, n), wrapI(row, n)), 0).r;
  }
  // Absorb: Dirichlet φ=0 at ghost cells — potential vanishes at open boundary.
  if (params.boundaryMode == 1u && (col < 0 || col >= n || row < 0 || row >= n)) {
    return 0.0;
  }
  // Reflect: Neumann (clamp) — zero gradient at solid wall.
  return textureLoad(potentialIn, vec2i(clamp(col, 0, n - 1), clamp(row, 0, n - 1)), 0).r;
}

@compute @workgroup_size(8, 8)
fn jacobiGravityStep(@builtin(global_invocation_id) id: vec3u) {
  let col = i32(id.x);
  let row = i32(id.y);
  let n   = i32(params.gridSize);
  if (col >= n || row >= n) { return; }

  let left   = samplePotential(col - 1, row);
  let right  = samplePotential(col + 1, row);
  let bottom = samplePotential(col, row - 1);
  let top    = samplePotential(col, row + 1);
  let sw     = samplePotential(col - 1, row - 1);
  let se     = samplePotential(col + 1, row - 1);
  let nw     = samplePotential(col - 1, row + 1);
  let ne     = samplePotential(col + 1, row + 1);
  let rho    = textureLoad(densityIn, vec2i(col, row), 0).r;

  // Mehrstellen 9-point isotropic Laplacian, source = density.
  let phi = (4.0 * (left + right + bottom + top) + (sw + se + nw + ne) - 6.0 * rho) / 20.0;
  textureStore(potentialOut, vec2i(col, row), vec4f(phi, 0.0, 0.0, 0.0));
}

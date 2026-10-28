// Semi-Lagrangian advection for the scalar density field.
// Density moves ONLY under gravitational drift: -∇φ * gravStrength.
// Fluid velocity is intentionally excluded — divergence-free NS advection is
// volume-preserving and cannot cause clumping. Only the gravity drift, which is
// not constrained by ∇·v = 0, allows density to converge toward mass centers.

struct DensityAdvectParams {
  gridSize:     u32,
  boundaryMode: u32,
  deltaTime:    f32,
  damping:      f32,  // pow(densityDamping, deltaTime) — pre-computed on JS side
  gravStrength: f32,
  pad0:         u32,
  pad1:         u32,
  pad2:         u32,
}

@group(0) @binding(0) var velocityTexture:  texture_2d<f32>; // unused — kept for API consistency
@group(0) @binding(1) var densityIn:        texture_2d<f32>;
@group(0) @binding(2) var densityOut:       texture_storage_2d<r32float, write>;
@group(0) @binding(3) var<uniform> params:  DensityAdvectParams;
@group(0) @binding(4) var potentialTexture: texture_2d<f32>;

fn wrapI(c: i32, n: i32) -> i32 {
  return ((c % n) + n) % n;
}

fn sampleDensityAt(col: i32, row: i32) -> f32 {
  let n = i32(params.gridSize);
  if (params.boundaryMode == 0u) {
    return textureLoad(densityIn, vec2i(wrapI(col, n), wrapI(row, n)), 0).r;
  }
  if (col < 0 || col >= n || row < 0 || row >= n) { return 0.0; }
  return textureLoad(densityIn, vec2i(col, row), 0).r;
}

fn sampleDensityBilinear(posX: f32, posY: f32) -> f32 {
  let x0 = i32(floor(posX));
  let y0 = i32(floor(posY));
  let fx  = fract(posX);
  let fy  = fract(posY);
  let s00 = sampleDensityAt(x0,     y0);
  let s10 = sampleDensityAt(x0 + 1, y0);
  let s01 = sampleDensityAt(x0,     y0 + 1);
  let s11 = sampleDensityAt(x0 + 1, y0 + 1);
  return mix(mix(s00, s10, fx), mix(s01, s11, fx), fy);
}

fn samplePotential(col: i32, row: i32) -> f32 {
  let n = i32(params.gridSize);
  if (params.boundaryMode == 0u) {
    return textureLoad(potentialTexture, vec2i(wrapI(col, n), wrapI(row, n)), 0).r;
  }
  return textureLoad(potentialTexture, vec2i(clamp(col, 0, n - 1), clamp(row, 0, n - 1)), 0).r;
}

@compute @workgroup_size(8, 8)
fn advectDensity(@builtin(global_invocation_id) id: vec3u) {
  let col = i32(id.x);
  let row = i32(id.y);
  let n   = i32(params.gridSize);
  if (col >= n || row >= n) { return; }

  // Gravitational drift velocity: g = -∇φ (central differences).
  // Density "falls" toward lower potential (toward mass concentrations).
  let dPhiDx = (samplePotential(col + 1, row) - samplePotential(col - 1, row)) * 0.5;
  let dPhiDy = (samplePotential(col, row + 1) - samplePotential(col, row - 1)) * 0.5;
  let gravityDrift = -vec2f(dPhiDx, dPhiDy) * params.gravStrength;

  let tracedCol = f32(col) - gravityDrift.x * params.deltaTime;
  let tracedRow = f32(row) - gravityDrift.y * params.deltaTime;

  let advected = sampleDensityBilinear(tracedCol, tracedRow) * params.damping;
  textureStore(densityOut, vec2i(col, row), vec4f(max(0.0, advected), 0.0, 0.0, 0.0));
}

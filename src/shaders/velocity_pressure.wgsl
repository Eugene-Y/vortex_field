// One Jacobi iteration of the 9-point Mehrstellen isotropic pressure solve.
// Reads pressureIn + divergence, writes one updated pressure estimate to pressureOut.
// Run N times per frame with ping-pong buffers; N controls incompressibility.

struct BaseParams {
  gridSize:     u32,
  boundaryMode: u32,
  pad0:         u32,
  pad1:         u32,
}

@group(0) @binding(0) var pressureIn:   texture_2d<f32>;
@group(0) @binding(1) var divergenceIn: texture_2d<f32>;
@group(0) @binding(2) var pressureOut:  texture_storage_2d<r32float, write>;
@group(0) @binding(3) var<uniform> params: BaseParams;

fn wrapI(c: i32, n: i32) -> i32 {
  return ((c % n) + n) % n;
}

fn samplePressure(col: i32, row: i32) -> f32 {
  let n = i32(params.gridSize);
  if (params.boundaryMode == 0u) {
    return textureLoad(pressureIn, vec2i(wrapI(col, n), wrapI(row, n)), 0).r;
  }
  // Absorb: Dirichlet p=0 at ghost cells — open boundary, fluid exits freely.
  if (params.boundaryMode == 1u && (col < 0 || col >= n || row < 0 || row >= n)) {
    return 0.0;
  }
  // Reflect: Neumann (clamp) — zero gradient at solid wall.
  return textureLoad(pressureIn, vec2i(clamp(col, 0, n - 1), clamp(row, 0, n - 1)), 0).r;
}

@compute @workgroup_size(8, 8)
fn jacobiPressureStep(@builtin(global_invocation_id) id: vec3u) {
  let col = i32(id.x);
  let row = i32(id.y);
  let n   = i32(params.gridSize);
  if (col >= n || row >= n) { return; }

  let left   = samplePressure(col - 1, row);
  let right  = samplePressure(col + 1, row);
  let bottom = samplePressure(col, row - 1);
  let top    = samplePressure(col, row + 1);
  let sw     = samplePressure(col - 1, row - 1);
  let se     = samplePressure(col + 1, row - 1);
  let nw     = samplePressure(col - 1, row + 1);
  let ne     = samplePressure(col + 1, row + 1);
  let div    = textureLoad(divergenceIn, vec2i(col, row), 0).r;

  // Mehrstellen 9-point isotropic Laplacian: cardinal ×4, diagonals ×1.
  // Reduces the directional bias that causes diagonal artifacts with the 5-point stencil.
  let pressure = (4.0 * (left + right + bottom + top) + (sw + se + nw + ne) - 6.0 * div) / 20.0;
  textureStore(pressureOut, vec2i(col, row), vec4f(pressure, 0.0, 0.0, 0.0));
}

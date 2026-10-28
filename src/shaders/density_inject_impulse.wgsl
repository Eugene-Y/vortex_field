// Adds a Gaussian-weighted density blob at a given UV position.
// Scalar injection — no direction, mass is deposited symmetrically.

struct DensityImpulseParams {
  gridSize:  u32,
  boundaryMode: u32,
  radius:    f32,    // in UV [0, 1] units (= grid cells / gridSize)
  strength:  f32,
  position:  vec2f,  // in UV [0, 1]
  pad0:      f32,
  pad1:      f32,
}

@group(0) @binding(0) var densityIn:  texture_2d<f32>;
@group(0) @binding(1) var densityOut: texture_storage_2d<r32float, write>;
@group(0) @binding(2) var<uniform> params: DensityImpulseParams;

fn gaussianDensity(uv: vec2f) -> f32 {
  var offset = uv - params.position;
  // Periodic minimum-image: fold offset into [-0.5, 0.5] in UV space.
  if (params.boundaryMode == 0u) { offset -= round(offset); }
  let distSq   = dot(offset, offset);
  let radiusSq = params.radius * params.radius;
  return params.strength * exp(-distSq / radiusSq);
}

@compute @workgroup_size(8, 8)
fn injectDensityImpulse(@builtin(global_invocation_id) id: vec3u) {
  let col = i32(id.x);
  let row = i32(id.y);
  let n   = i32(params.gridSize);
  if (col >= n || row >= n) { return; }

  // Absorb: boundary cells act as ghost cells — skip injection there.
  if (params.boundaryMode == 1u && (col == 0 || col == n - 1 || row == 0 || row == n - 1)) {
    textureStore(densityOut, vec2i(col, row), textureLoad(densityIn, vec2i(col, row), 0));
    return;
  }

  let uv       = (vec2f(f32(col), f32(row)) + 0.5) / f32(n);
  let existing = textureLoad(densityIn, vec2i(col, row), 0).r;
  textureStore(densityOut, vec2i(col, row), vec4f(max(0.0, existing + gaussianDensity(uv)), 0.0, 0.0, 0.0));
}

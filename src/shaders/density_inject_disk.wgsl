// Injects a filled disk of density in a single pass.
// Gaussian radial falloff for a smooth boundary that avoids divergence artifacts.

struct DensityDiskParams {
  gridSize:     u32,
  boundaryMode: u32,
  radius:       f32,   // in UV [0, 1] units
  strength:     f32,
  center:       vec2f, // in UV [0, 1]
  pad0:         f32,
  pad1:         f32,
}

@group(0) @binding(0) var densityIn:  texture_2d<f32>;
@group(0) @binding(1) var densityOut: texture_storage_2d<r32float, write>;
@group(0) @binding(2) var<uniform> params: DensityDiskParams;

@compute @workgroup_size(8, 8)
fn injectDensityDisk(@builtin(global_invocation_id) id: vec3u) {
  let col = i32(id.x);
  let row = i32(id.y);
  let n   = i32(params.gridSize);
  if (col >= n || row >= n) { return; }

  if (params.boundaryMode == 1u && (col == 0 || col == n - 1 || row == 0 || row == n - 1)) {
    textureStore(densityOut, vec2i(col, row), textureLoad(densityIn, vec2i(col, row), 0));
    return;
  }

  let uv       = (vec2f(f32(col), f32(row)) + 0.5) / f32(n);
  let offset   = uv - params.center;
  let dist     = length(offset);
  let existing = textureLoad(densityIn, vec2i(col, row), 0).r;

  var result = existing;
  if (dist <= params.radius) {
    let t    = dist / params.radius;
    let edge = exp(-3.5 * t * t); // Gaussian falloff avoids sharp boundary divergence
    result  += params.strength * edge;
  }

  textureStore(densityOut, vec2i(col, row), vec4f(max(0.0, result), 0.0, 0.0, 0.0));
}

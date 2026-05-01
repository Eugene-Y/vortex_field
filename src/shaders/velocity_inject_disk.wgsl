// Injects a filled disk of velocity (spin / explode / implode) in a single pass.
// Gaussian radial falloff avoids a sharp boundary that would create a divergence ring.

struct DiskParams {
  gridSize:     u32,
  boundaryMode: u32,
  radius:       f32,   // in UV [0, 1] units
  strength:     f32,
  center:       vec2f, // in UV [0, 1]
  mode:         u32,   // 0 = spin (CCW), 1 = explode, 2 = implode
  pad0:         u32,
}

@group(0) @binding(0) var velocityIn:  texture_2d<f32>;
@group(0) @binding(1) var velocityOut: texture_storage_2d<rg32float, write>;
@group(0) @binding(2) var<uniform> params: DiskParams;

// Returns the velocity direction for a point at offset from the disk centre.
fn diskVelocityDirection(offset: vec2f) -> vec2f {
  let dist = length(offset);
  if (dist < 1e-6) { return vec2f(0.0); }
  let radial  =  offset / dist;
  let tangent = vec2f(-radial.y, radial.x); // CCW perpendicular
  if (params.mode == 0u) { return tangent; }
  if (params.mode == 1u) { return radial; }
  return -radial;
}

@compute @workgroup_size(8, 8)
fn injectDisk(@builtin(global_invocation_id) id: vec3u) {
  let col = i32(id.x);
  let row = i32(id.y);
  let n   = i32(params.gridSize);
  if (col >= n || row >= n) { return; }

  // Absorb: skip boundary ghost cells.
  if (params.boundaryMode == 1u && (col == 0 || col == n - 1 || row == 0 || row == n - 1)) {
    textureStore(velocityOut, vec2i(col, row), textureLoad(velocityIn, vec2i(col, row), 0));
    return;
  }

  let uv       = (vec2f(f32(col), f32(row)) + 0.5) / f32(n);
  let offset   = uv - params.center;
  let dist     = length(offset);
  let existing = textureLoad(velocityIn, vec2i(col, row), 0).xy;

  var result = existing;
  if (dist <= params.radius) {
    let t    = dist / params.radius;
    let edge = exp(-3.5 * t * t); // Gaussian falloff; avoids divergence spike at boundary
    result  += diskVelocityDirection(offset) * params.strength * edge;
  }

  textureStore(velocityOut, vec2i(col, row), vec4f(result, 0.0, 0.0));
}

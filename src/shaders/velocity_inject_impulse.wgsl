// Adds a Gaussian-weighted velocity impulse at a given UV position.
// Used for mouse stroke injection.

struct ImpulseParams {
  gridSize:         u32,
  boundaryMode:     u32,
  impulseRadius:    f32,   // in UV [0, 1] units (= grid cells / gridSize)
  impulseStrength:  f32,
  impulsePosition:  vec2f, // in UV [0, 1]
  impulseDirection: vec2f, // unit vector
}

@group(0) @binding(0) var velocityIn:  texture_2d<f32>;
@group(0) @binding(1) var velocityOut: texture_storage_2d<rg32float, write>;
@group(0) @binding(2) var<uniform> params: ImpulseParams;

fn gaussianImpulse(uv: vec2f) -> vec2f {
  var offset = uv - params.impulsePosition;
  // Periodic minimum-image: fold offset into [-0.5, 0.5] in UV space.
  if (params.boundaryMode == 0u) { offset -= round(offset); }
  let distSq   = dot(offset, offset);
  let radiusSq = params.impulseRadius * params.impulseRadius;
  return params.impulseDirection * params.impulseStrength * exp(-distSq / radiusSq);
}

@compute @workgroup_size(8, 8)
fn injectGaussianImpulse(@builtin(global_invocation_id) id: vec3u) {
  let col = i32(id.x);
  let row = i32(id.y);
  let n   = i32(params.gridSize);
  if (col >= n || row >= n) { return; }

  // Absorb: boundary cells act as ghost cells — skip injection there.
  if (params.boundaryMode == 1u && (col == 0 || col == n - 1 || row == 0 || row == n - 1)) {
    textureStore(velocityOut, vec2i(col, row), textureLoad(velocityIn, vec2i(col, row), 0));
    return;
  }

  // Cell centre in UV space: (col + 0.5) / n
  let uv       = (vec2f(f32(col), f32(row)) + 0.5) / f32(n);
  let existing = textureLoad(velocityIn, vec2i(col, row), 0).xy;
  textureStore(velocityOut, vec2i(col, row), vec4f(existing + gaussianImpulse(uv), 0.0, 0.0));
}

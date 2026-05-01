// Applies vorticity confinement force: re-energises vortex cores that numerical
// diffusion would otherwise smooth away.
// Force = ε × ω × (η.y, −η.x) × dt, where η = normalised ∇|ω|.

struct ConfinementParams {
  gridSize:  u32,
  pad0:      u32,
  strength:  f32,
  deltaTime: f32,
}

@group(0) @binding(0) var velocityIn: texture_2d<f32>;
@group(0) @binding(1) var curlIn:     texture_2d<f32>;
@group(0) @binding(2) var velocityOut: texture_storage_2d<rg32float, write>;
@group(0) @binding(3) var<uniform> params: ConfinementParams;

fn sampleCurl(col: i32, row: i32) -> f32 {
  let n = i32(params.gridSize);
  return textureLoad(curlIn, vec2i(clamp(col, 0, n - 1), clamp(row, 0, n - 1)), 0).r;
}

@compute @workgroup_size(8, 8)
fn applyVorticityConfinement(@builtin(global_invocation_id) id: vec3u) {
  let col = i32(id.x);
  let row = i32(id.y);
  let n   = i32(params.gridSize);
  if (col >= n || row >= n) { return; }

  let curl = sampleCurl(col, row);
  // Gradient of |curl| magnitude via central differences.
  let grad = vec2f(
    abs(sampleCurl(col + 1, row)) - abs(sampleCurl(col - 1, row)),
    abs(sampleCurl(col, row + 1)) - abs(sampleCurl(col, row - 1)),
  ) * 0.5;

  let len  = length(grad) + 1e-5;
  let eta  = grad / len;  // unit vector pointing toward stronger curl
  let force = params.strength * curl * vec2f(eta.y, -eta.x);

  let velocity = textureLoad(velocityIn, vec2i(col, row), 0).xy + force * params.deltaTime;
  textureStore(velocityOut, vec2i(col, row), vec4f(velocity, 0.0, 0.0));
}

// WebGPU render pipeline for Field B (rotation field).
// Reads from the f32 rotation output buffer, applies same color encoding
// as render.frag rotation mode: orange=CCW, blue=CW, Reinhard tonemapping.

struct RenderParams {
  gridSize:            u32,
  rotationToneMidpoint: f32,
  colorPositive:       vec3f,  // CCW — orange by default
  colorNegative:       vec3f,  // CW  — blue  by default
}

@group(0) @binding(0) var<storage, read> rotationOutput: array<f32>;
@group(0) @binding(1) var<uniform>       params:         RenderParams;

struct VertexOut {
  @builtin(position) position: vec4f,
  @location(0)       uv:       vec2f,
}

@vertex
fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> VertexOut {
  // Full-screen triangle strip: indices 0-3 → corners (-1,-1) (1,-1) (-1,1) (1,1)
  let x = f32(vertexIndex & 1u) * 2.0 - 1.0;
  let y = f32((vertexIndex >> 1u) & 1u) * 2.0 - 1.0;
  var out: VertexOut;
  out.position = vec4f(x, y, 0.0, 1.0);
  out.uv = vec2f(x * 0.5 + 0.5, y * 0.5 + 0.5);
  return out;
}

fn mapRotationToColor(rotation: f32) -> vec3f {
  let absolute  = abs(rotation);
  let magnitude = absolute / (absolute + params.rotationToneMidpoint);
  if (rotation > 0.0) {
    return params.colorPositive * magnitude;
  } else {
    return params.colorNegative * magnitude;
  }
}

@fragment
fn fragmentMain(in: VertexOut) -> @location(0) vec4f {
  let col     = u32(in.uv.x * f32(params.gridSize));
  let row     = u32(in.uv.y * f32(params.gridSize));
  let clamped = vec2u(min(col, params.gridSize - 1u), min(row, params.gridSize - 1u));
  let index   = clamped.y * params.gridSize + clamped.x;
  let rotation = rotationOutput[index];
  return vec4f(mapRotationToColor(rotation), 1.0);
}

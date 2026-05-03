// WebGPU render pipeline for Field B (rotation field).
// Reads from the f32 rotation output buffer, applies same color encoding
// as render.frag rotation mode: orange=CCW, blue=CW, Reinhard tonemapping.
//
// Values are normalized by the smoothed mean absolute energy before tone-mapping
// so the brightness slider remains consistent regardless of pair distance / delta.

struct RenderParams {
  gridSize:             u32,   // 0
  rotationToneMidpoint: f32,   // 4
  autoNormalize:        u32,   // 8   1 = divide by smoothed mean; 0 = raw values
  _pad1:                f32,   // 12
  colorPositive:        vec3f, // 16  CCW — orange by default
  colorNegative:        vec3f, // 32  CW  — blue  by default
}                               // 48 bytes

@group(0) @binding(0) var<storage, read> rotationOutput: array<f32>;
@group(0) @binding(1) var<uniform>       params:         RenderParams;
@group(0) @binding(2) var<storage, read> smoothedEnergy: array<f32>; // [0] = EMA of Σ|value|

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

fn mapRotationToColor(rotation: f32, meanAbsValue: f32) -> vec3f {
  // Normalize by mean absolute value so the tone midpoint is stable across
  // changes to pair distance/delta. Guard against division by zero.
  let normalized = rotation / max(meanAbsValue, 1e-8);
  let absolute   = abs(normalized);
  let magnitude  = absolute / (absolute + params.rotationToneMidpoint);
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

  let rotation     = rotationOutput[index];
  let totalCells   = f32(params.gridSize * params.gridSize);
  let meanAbsValue = select(1.0, smoothedEnergy[0] / totalCells, params.autoNormalize == 1u);

  return vec4f(mapRotationToColor(rotation, meanAbsValue), 1.0);
}

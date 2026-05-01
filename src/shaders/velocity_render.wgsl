// WebGPU render pipeline for Field A (velocity field).
// Maps velocity magnitude → brightness (Reinhard tonemapping),
// direction → hue (HSV encoding). Orange/warm = fast, black = still.

struct RenderParams {
  gridSize:             u32,
  pad0:                 u32,
  velocityToneMidpoint: f32,
  pad1:                 f32,
}

@group(0) @binding(0) var velocityTexture: texture_2d<f32>;
@group(0) @binding(1) var<uniform> params: RenderParams;

struct VertexOut {
  @builtin(position) position: vec4f,
  @location(0)       uv:       vec2f,
}

@vertex
fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> VertexOut {
  // Full-screen triangle strip; indices 0-3 → corners (-1,-1) (1,-1) (-1,1) (1,1).
  let x = f32(vertexIndex & 1u) * 2.0 - 1.0;
  let y = f32((vertexIndex >> 1u) & 1u) * 2.0 - 1.0;
  var out: VertexOut;
  out.position = vec4f(x, y, 0.0, 1.0);
  // uv.y=0 at NDC y=-1 (screen bottom) = physical row 0 (y-up convention).
  out.uv = vec2f(x * 0.5 + 0.5, y * 0.5 + 0.5);
  return out;
}

fn hsvToRgb(hsv: vec3f) -> vec3f {
  let h = hsv.x * 6.0;
  let s = hsv.y;
  let v = hsv.z;
  let c = v * s;
  let x = c * (1.0 - abs(h % 2.0 - 1.0));
  let m = v - c;
  var rgb: vec3f;
  if      (h < 1.0) { rgb = vec3f(c, x, 0.0); }
  else if (h < 2.0) { rgb = vec3f(x, c, 0.0); }
  else if (h < 3.0) { rgb = vec3f(0.0, c, x); }
  else if (h < 4.0) { rgb = vec3f(0.0, x, c); }
  else if (h < 5.0) { rgb = vec3f(x, 0.0, c); }
  else              { rgb = vec3f(c, 0.0, x); }
  return rgb + m;
}

fn mapVelocityToColor(velocity: vec2f) -> vec3f {
  let magnitude  = length(velocity);
  let brightness = magnitude / (magnitude + params.velocityToneMidpoint);
  let angle      = atan2(velocity.y, velocity.x);
  let hue        = (angle + 3.14159265) / (2.0 * 3.14159265);
  return hsvToRgb(vec3f(hue, 0.8, brightness));
}

@fragment
fn fragmentMain(in: VertexOut) -> @location(0) vec4f {
  let col     = u32(in.uv.x * f32(params.gridSize));
  let row     = u32(in.uv.y * f32(params.gridSize));
  let clamped = vec2u(min(col, params.gridSize - 1u), min(row, params.gridSize - 1u));
  let velocity = textureLoad(velocityTexture, vec2i(clamped), 0).xy;
  return vec4f(mapVelocityToColor(velocity), 1.0);
}

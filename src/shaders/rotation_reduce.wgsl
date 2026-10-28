// Conversion pass: turns the 64-bit fixed-point accumulation into the flat f32 output.
// One thread per grid cell; the high word is signed, the low word unsigned.

struct ReduceParams {
  gridSize: u32,
  _pad0:    u32,
  _pad1:    u32,
  _pad2:    u32,
}

@group(0) @binding(0) var<storage, read>       accumulation:   array<u32>;
@group(0) @binding(1) var<storage, read_write> rotationOutput: array<f32>;
@group(0) @binding(2) var<uniform>             params:         ReduceParams;

override FIXED_POINT_SCALE: f32;

const TWO_POW_16 = 65536.0;

// Horner over 16-bit digits: for small negative sums (high word −1, low word near 2³²)
// the cancellation happens exactly instead of rounding the low word to f32 first.
fn fixed64ToFloat(lowWord: u32, highWord: u32) -> f32 {
  let highDigits = f32(bitcast<i32>(highWord));
  let upperDigit = f32(lowWord >> 16u);
  let lowerDigit = f32(lowWord & 0xFFFFu);
  return ((highDigits * TWO_POW_16 + upperDigit) * TWO_POW_16 + lowerDigit) / FIXED_POINT_SCALE;
}

@compute @workgroup_size(64)
fn reduceBuffers(@builtin(global_invocation_id) globalId: vec3<u32>) {
  let cellIndex = globalId.x;
  if (cellIndex >= params.gridSize * params.gridSize) { return; }
  rotationOutput[cellIndex] = fixed64ToFloat(accumulation[2u * cellIndex], accumulation[2u * cellIndex + 1u]);
}

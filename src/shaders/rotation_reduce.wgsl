// Reduction pass: sums K float-valued accumulation buffers into a single f32 output.
// Each buffer stores f32 bit-patterns in i32 slots (CAS float atomic convention).
// One thread per grid cell — reads K values, bit-casts to f32, sums, writes.

struct ReduceParams {
  gridSize:    u32,
  bufferCount: u32,  // K
  _pad0:       u32,
  _pad1:       u32,
}

@group(0) @binding(0) var<storage, read>       rotationBuffers: array<i32>;
@group(0) @binding(1) var<storage, read_write> rotationOutput:  array<f32>;
@group(0) @binding(2) var<uniform>             params:          ReduceParams;

@compute @workgroup_size(64)
fn reduceBuffers(@builtin(global_invocation_id) globalId: vec3<u32>) {
  let cellIndex  = globalId.x;
  let totalCells = params.gridSize * params.gridSize;
  if (cellIndex >= totalCells) { return; }

  var sum: f32 = 0.0;
  for (var k: u32 = 0u; k < params.bufferCount; k++) {
    // bit-cast: the CAS float atomic stores f32 bit-pattern as i32.
    // bitcast<f32>(0) == 0.0, so cleared cells contribute nothing.
    sum += bitcast<f32>(rotationBuffers[k * totalCells + cellIndex]);
  }

  rotationOutput[cellIndex] = sum;
}

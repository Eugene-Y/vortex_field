// Energy smoothing pass for Field B auto-brightness normalization.
//
// A single workgroup of 64 threads partitions the output buffer into equal strips,
// sums |value| per strip, then reduces via a binary tree to yield the total absolute
// energy. An exponential moving average is applied in-place.
//
// Cold-start handling: when the EMA buffer is zero (empty field) and non-zero energy
// arrives, the EMA is initialized directly rather than smoothing from 0, preventing
// a slow-to-converge bright transient on first velocity injection.

struct EnergyParams {
  totalCells: u32,
  emaAlpha:   f32,
  _pad0:      f32,
  _pad1:      f32,
}

@group(0) @binding(0) var<storage, read>       rotationOutput: array<f32>;
@group(0) @binding(1) var<storage, read_write> smoothedEnergy: array<f32>; // [0] = EMA
@group(0) @binding(2) var<uniform>             params:         EnergyParams;

// Sample every ENERGY_STRIDE-th cell — gives a 16× cheaper pass while remaining
// representative since the rotation field is spatially correlated.
const ENERGY_STRIDE: u32 = 16u;

var<workgroup> partialSums: array<f32, 64>;

@compute @workgroup_size(64)
fn computeSmoothedEnergy(@builtin(local_invocation_id) lid: vec3<u32>) {
  // Each thread covers a strided strip of the sampled elements.
  let sampleCount = (params.totalCells + ENERGY_STRIDE - 1u) / ENERGY_STRIDE;
  let chunkSize   = (sampleCount + 63u) / 64u;
  let start       = lid.x * chunkSize;
  let end         = min(start + chunkSize, sampleCount);

  var localSum = 0.0;
  for (var i = start; i < end; i++) {
    localSum += abs(rotationOutput[i * ENERGY_STRIDE]);
  }
  // Scale back up to approximate the full sum.
  localSum *= f32(ENERGY_STRIDE);
  partialSums[lid.x] = localSum;
  workgroupBarrier();

  // Binary tree reduction within workgroup. workgroupBarrier() is outside the
  // conditional so it remains in uniform control flow on all iterations.
  for (var stride = 32u; stride >= 1u; stride >>= 1u) {
    if (lid.x < stride) {
      partialSums[lid.x] += partialSums[lid.x + stride];
    }
    workgroupBarrier();
  }

  if (lid.x == 0u) {
    let totalAbsEnergy = partialSums[0];
    let prev           = smoothedEnergy[0];
    if (prev > 1e-10) {
      // Normal EMA update.
      smoothedEnergy[0] = params.emaAlpha * totalAbsEnergy + (1.0 - params.emaAlpha) * prev;
    } else {
      // Cold start: initialize directly so the first frame with velocity shows
      // correct normalization rather than 50× over-brightness.
      smoothedEnergy[0] = totalAbsEnergy;
    }
  }
}

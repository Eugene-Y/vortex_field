# WebGPU Rotation Field — Migration Plan

## Goal

Replace `PairRotationKernel` (WebGL vertex-shader trick) with a WebGPU compute shader.
Keep Navier-Stokes entirely in WebGL. Only the rotation field (Field B) moves to WebGPU.

Target: GRID_SIZE=512 at ≥4fps.

## Why

Current WebGL approach draws `GRID_SIZE^4` points via `gl.drawArrays(POINTS, 0, N^4)`.
At N=512 that is 68 billion vertex invocations submitted even when pairRange discards most.
WebGPU compute dispatches only `N^2` threads; each iterates only over j within pairRange
radius — actual work is O(pairRange^2 × N^4), no idle vertex submissions.

## Algorithm

Each compute thread = one cell i.
Inner loop: all cells j in pairRange radius where j > i (unique pairs only).
Per valid pair: compute instantaneous rotation center, compute ω, atomicAdd to
one of K private accumulation buffers (selected by workgroup_id % K).
Final pass: sum K buffers into one rotation texture for rendering.

Fixed-point atomics: ω × FIXED_POINT_SCALE → i32 → atomicAdd.
Divide by FIXED_POINT_SCALE in render shader.

K=16 buffers → contention reduced 16×. Memory: 16 × 512×512×4 = 16MB.

## Data Flow Per Frame

1. WebGL: Navier-Stokes step → velocity WebGL texture (unchanged)
2. `VelocityBridge`: gl.readPixels → Float32Array → device.queue.writeTexture
3. WebGPU compute pass: accumulate rotation into K atomic i32 storage buffers
4. WebGPU reduction pass: sum K buffers → f32 rotation texture
5. WebGPU render pass: rotation texture → canvas-rotation (Field B display)
6. WebGL render pass: velocity texture → canvas-main (Field A display, unchanged)

## Files

### New
- `src/gpu/WebGPUDevice.js`          — request adapter/device, check support, expose device+queue
- `src/gpu/VelocityBridge.js`        — gl.readPixels each frame → upload to WebGPU texture
- `src/simulation/WebGPURotationField.js` — orchestrates bridge + compute + render; same external
                                            interface as old RotationField (recomputeFrom, dispose)
- `src/shaders/rotation_compute.wgsl`  — compute shader: N^2 threads, inner loop over neighbors,
                                         fixed-point atomicAdd into K buffers
- `src/shaders/rotation_reduce.wgsl`   — reduction pass: sum K buffers → f32 texture
- `src/shaders/rotation_render.wgsl`   — WebGPU render pipeline: f32 texture → canvas-rotation
                                         (same color logic as current render.frag rotation mode)

### Modified
- `index.html`          — add <canvas id="canvas-rotation">, CSS flex layout side by side,
                          remove DISPLAY_GAP from WebGL canvas (gap handled by CSS margin)
- `src/main.js`         — init WebGPU alongside WebGL; canvas-main = Field A only (half width);
                          use WebGPURotationField instead of RotationField;
                          remove renderRotationField call from renderFields()
- `src/rendering/FieldRenderer.js` — remove renderRotationField method (no longer used)

### Deleted
- `src/simulation/PairRotationKernel.js`
- `src/simulation/RotationField.js`
- `src/shaders/rotation_accumulate.vert`
- `src/shaders/rotation_accumulate.frag`

## Constants

```js
const FIXED_POINT_SCALE = 1_000_000; // ω range assumed < ~2000 → no i32 overflow
const ACCUMULATION_BUFFERS = 16;     // K private buffers for contention reduction
const WORKGROUP_SIZE = 64;           // threads per compute workgroup
```

## Compute Shader Sketch (rotation_compute.wgsl)

```wgsl
struct Params {
  gridSize:           u32,
  accumulationScale:  f32,
  parallelThreshold:  f32,
  pairRange:          f32,
  bufferIndex:        u32,   // which of K buffers this dispatch writes to
  fixedPointScale:    f32,
}

@group(0) @binding(0) var velocityTexture: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> rotationBuffer: array<atomic<i32>>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn computeRotation(@builtin(global_invocation_id) id: vec3<u32>) {
  let indexA = id.x;
  let totalCells = params.gridSize * params.gridSize;
  if (indexA >= totalCells) { return; }

  let colA = i32(indexA % params.gridSize);
  let rowA = i32(indexA / params.gridSize);
  let posA = vec2f(f32(colA), f32(rowA));
  let velA = textureLoad(velocityTexture, vec2i(colA, rowA), 0).xy;
  if (length(velA) < 1e-6) { return; }

  let radius = params.pairRange * f32(params.gridSize) * 0.5;
  let iRadius = i32(ceil(radius));
  let gSize = f32(params.gridSize);

  for (var dRow = -iRadius; dRow <= iRadius; dRow++) {
    for (var dCol = -iRadius; dCol <= iRadius; dCol++) {
      // periodic neighbor position
      let colB = ((colA + dCol) % i32(params.gridSize) + i32(params.gridSize)) % i32(params.gridSize);
      let rowB = ((rowA + dRow) % i32(params.gridSize) + i32(params.gridSize)) % i32(params.gridSize);
      let indexB = u32(rowB) * params.gridSize + u32(colB);

      if (indexB <= indexA) { continue; }  // unique pairs only

      let pairDisp = periodicDisplacement(posA, vec2f(f32(colB), f32(rowB)), gSize);
      let distSq = dot(pairDisp, pairDisp);
      let outerR = radius;
      if (distSq > outerR * outerR) { continue; }

      let velB = textureLoad(velocityTexture, vec2i(colB, rowB), 0).xy;
      // ... compute center, omega, atomicAdd
    }
  }
}
```

## Rollout Steps

1. [x] Create branch webgpu-rotation
2. [ ] WebGPUDevice.js — init, feature detect
3. [ ] VelocityBridge.js — readPixels → WebGPU texture
4. [ ] rotation_compute.wgsl — compute kernel
5. [ ] rotation_reduce.wgsl — K buffers → f32 texture
6. [ ] rotation_render.wgsl — WebGPU render to canvas
7. [ ] WebGPURotationField.js — orchestrator
8. [ ] index.html — second canvas, CSS layout
9. [ ] main.js — wire up WebGPU path
10. [ ] FieldRenderer.js — remove dead renderRotationField
11. [ ] Delete old rotation files
12. [ ] Test at GRID_SIZE=64, 128, 256, 512

## Fallback

If WebGPU not available (Firefox, old browser): show error message pointing to Chrome.
No WebGL fallback needed — the old code is on main branch.

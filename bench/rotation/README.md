# Field B performance research

Benchmark harness and findings for the rotation-field accumulation pass
(`src/shaders/rotation_compute.wgsl`), the dominant GPU cost of the app.

## Harness

```
deno run --unstable-webgpu --allow-read --allow-write bench/rotation/capture_simulated_fields.js 256
deno run --unstable-webgpu --allow-read bench/rotation/run_deno.js medium,large vortices,simulatedSpin production,wgCacheSpans
```

The same plan runs in a browser at `bench/rotation/?configs=…&scenarios=…&variants=…`
(needed for `subgroups`, which Deno's wgpu does not expose).

- `VelocityScenarios.js` — synthetic fields: `vortices` (14 Lamb–Oseen vortices), `turbulent`,
  `solidBody` (every pair shares one center — worst-case contention), `whiteNoise`
  (centers scattered — no contention). `simulated*` fields are captured from the production
  Navier–Stokes step with app defaults (`simulatedSpin` = the startup “Disk — spin” after 600 steps).
- `RotationKernelGenerator.js` — builds kernels from orthogonal options so each idea is
  measured alone and combined.
- `reference/` — the original CAS shaders from `main`, the reference every variant is
  checked against. The `branch` variant runs the current `src/` shaders (this branch ports
  the best variant there).
- `RotationBenchmark.js` — one frame = clear + compute + reduce, as in the app. Reports GPU
  timestamps of the compute pass (median and min of 5 batches) and the error against the
  production output (`relL1 = Σ|x−ref| / Σ|ref|`, `relMax` relative to max |ref|).
- `results/` — raw logs. Apple M3 (MacBook Air, fanless): long runs throttle thermally, so
  compare variants within one block; `deno_round3.txt` has interleaved A/B repeats.

Configs (grid 256 unless noted): `appDefault` (pairDistance 0.0447 ± 0.003, 16 pairs/cell),
`medium` (0.1 ± 0.05, 512 pairs/cell), `large` (0.3 ± 0.1, 3092 pairs/cell),
`grid512`, `grid1024` (app defaults at 1024), `maskSmall`/`maskLarge` (focus mask).

## Where the time goes in production

1. **CAS float atomics under contention.** Field B exists to light up rotation centers, so
   pairs pile onto vortex cores by design. A CAS loop retries serially on a hot cell: solid
   body rotation, medium annulus — 670 ms/frame, while the same loop without accumulation takes
   1.5 ms. The 32 replica buffers only divide the hot spot by 32.
2. **Replica buffers.** K = 32 copies (8 MB at 256², 128 MB at 1024²) blow the cache footprint
   and cost clear + reduce traffic (~19 ms/frame at 1024²) while not fixing contention.
3. **Integer `%` in the inner loop** — four per pair (B wrap, target cell). GPUs have no
   integer divider; a conditional add gives the same result.
4. **Redundant per-pair math** — two `length()` and three divisions, recomputed wraps.
5. **Half the offset table is rejected** by `indexB <= indexA`.

## What works (all keep the semantics; output matches production to f32 noise)

| technique | why it is general, not tuned to one GPU |
|---|---|
| **64-bit fixed-point integer atomics** (two u32 words, carry from the returned low word) | native `atomicAdd` never retries; sums become exact and order-independent (bit-reproducible), whereas production's f32 sums depend on order (run to run 5e-5 on 1024² solid body) |
| **K = 1** | with retry-free atomics replicas only cost footprint and traffic |
| **workgroup-memory target cache** (direct-mapped, 256 slots, flushed once per workgroup) | absorbs hot spots with WebGPU-core features only; exact because it is integer |
| **closed-form ω**: `t = dot(δ, v_B) / cross(v_A, v_B)`, `ω_A = 1/t` | fewer ALU ops and divisions; the wrap-mode “nearest image” arm of production is kept exactly (`inverseTExact`) |
| **half-plane offsets as row spans** | each unordered pair once, no rejected iterations, B column increments, clipped once per span in Absorb/Reflect |
| **no integer modulo** in wraps | integer division is slow on every GPU vendor |
| **offset split over dispatch.y** for the focus mask | a small mask box cannot fill the GPU otherwise |

Compute-pass time, Deno (wgpu → Metal), medium annulus:

| field | production | fixed64 + spans | + workgroup cache |
|---|---:|---:|---:|
| vortices | 10.95 ms | 3.64 (×3.0) | 3.79 (×2.9) |
| simulatedSpin (app startup) | 23.16 | 4.60 (×5.0) | 3.86 (×6.0) |
| simulatedMixed | 10.44 | 4.14 (×2.5) | 4.02 (×2.6) |
| whiteNoise | 10.21 | 4.19 (×2.4) | 3.99 (×2.6) |
| solidBody | 706 | 26.7 (×26) | 2.02 (×350) |

Large annulus: ×3.0 / ×5.4 / ×3.3 / ×3.9 / ×354 with the cache. Grid 1024, app defaults:
vortices 151 → 64 ms, solid body 9.3 s → 15 ms. Focus mask: ×2.3–25 (best with the offset split). Chrome (Dawn → Metal)
shows the same ratios (`results/chrome_round1.txt`).

Remaining headroom: the pair loop without any accumulation runs at ~15 Gpair/s, the best
variant at ~9 Gpair/s on diffuse targets, so scattered atomics are still ~40 % of the time.

## What does not pay off

- **Subgroup aggregation** (merge same-target lanes before the atomic): slower than the
  workgroup cache everywhere, and slower than nothing on diffuse targets; needs an optional
  feature. Not worth it.
- **Workgroup cache at tiny annuli** (16 pairs/cell): reset/flush overhead dominates on
  diffuse fields (0.25 → 0.66 ms), but it still wins on the hot-spot fields the app starts with.
  Smaller caches (32–64 slots) narrow that gap; slot count could scale with the offset count.
- **Larger caches** (1024–2048 slots): occupancy loss and flush cost outweigh the hit rate.
- **Velocity in a storage buffer instead of a texture**, 8×8 tiled cell order alone,
  32-bit fixed point (no faster than 64-bit, and lossy).
- Structural alternatives considered and rejected: gather / Hough formulation (O(N³)
  rasterization of normals), sorting contributions by target (P × 8 bytes of memory),
  FFT / convolution (the target cell is a nonlinear function of both velocities).
  Reducing the pair count itself is only possible approximately (stride, temporal
  subsampling of offsets) and therefore changes the result.

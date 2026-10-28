# Field A performance

Profiling and equivalence tools for the Navier–Stokes step (`WebGPUNavierStokesStep.js`).

```
deno run --unstable-webgpu --allow-read bench/fluid/run_deno.js "gridSize=512"
deno run --unstable-webgpu --allow-read bench/fluid/compare_deno.js "boundary=2" [steps] [rounds]
```

- `FieldAProfiler.js` — GPU timestamps on every compute pass of a step, the GPU span from
  first to last pass (the difference is GPU idle time between passes) and CPU encode time.
  The browser page `bench/fluid/?<app query>` runs the same profile under Dawn and also
  times the render pass.
- `compare_deno.js` — runs `reference/` (the step and shaders as on `main` before this work)
  and the current `src/` step from the same flow, reports how many velocity values differ
  after N steps, and times both in alternating rounds.

## Findings (Apple M3, quiet GPU)

Before: the step was 80 compute passes, 75 of them single Jacobi pressure iterations. The
passes themselves took about half the GPU time, idle gaps between passes the other half
(Chrome: 8.6 ms of passes + 7.2 ms of gaps at 256²). Rendering Field A costs ~0.9 ms.

| change | result vs reference |
|---|---|
| one compute pass per step, cached bind groups | bit-identical |
| `wrapIntoGrid` instead of integer `%` (advection folds the integer cell index only) | bit-identical, all boundary modes |
| 4 Jacobi iterations per dispatch on a 32×32 workgroup tile | differs by a few ulp per step (same with 1 iteration per dispatch, so it is codegen, not the tiling); diverges chaotically over many steps like any rounding change |

Step time, steps back to back (Deno): 256² 4.7–5.3 → 1.8–2.0 ms, 512² ~19.5 → ~6 ms.
App in Chrome: 512² 40.6 → 60 fps (vsync), 1024² 7.5 → 13 fps.
2 and 4 iterations per dispatch perform alike; 6 is slower (thicker redundant border).

After the changes, a step at 256² is ~0.66 ms without pressure plus ~15 µs per Jacobi
iteration; pressure is still the larger part at 512² and above.

Known issue, also on `main`: with `?gravityExperiment=1&gravEnabled=1` the density-advect
bind group is invalid (5 entries, the auto layout has 4), so the whole step is dropped.

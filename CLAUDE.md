# Vortex Field — Project Brief

## Concept

Two coupled fields on a configurable grid (default 256×256, set via `GRID_SIZE`).

**Field A — Velocity field.**
A 2D fluid simulation (Navier-Stokes) on a grid. The user interacts with the mouse,
injecting impulses into the field. Energy propagates and decays according to configurable
parameters (damping, brush radius, brush strength). Advection uses Catmull-Rom bicubic
interpolation to suppress axis-aligned anisotropy artifacts. Explicit diffusion is omitted
entirely — numerical diffusion from the semi-Lagrangian advection scheme is sufficient.

**Field B — Instantaneous rotation field.**
Reset to zero every frame. For every pair of cells (i, j) in Field A, compute the
instantaneous center of rotation of those two cells relative to each other. The
signed magnitude of that relative rotation is accumulated into the cell of Field B
whose grid coordinate corresponds to that center of rotation. Sign encodes direction
(clockwise vs counter-clockwise). Magnitude encodes strength.

The GPU kernel evaluates only unique pairs (`indexA < indexB`) to avoid double-counting.
Angular velocity ω is computed from both cells independently and averaged, removing
any asymmetry introduced by the indexA < indexB selection. The interaction radius is controlled by `pairDistance` (annulus center) and
`distanceDelta` (annulus half-width), selecting pairs within
`[pairDistance − distanceDelta, pairDistance + distanceDelta] × gridSize/2` cells.

Pair displacement and center validity are boundary-mode-aware: in Wrap mode the
shortest torus path is used; in Absorb/Reflect modes the direct vector is used and
pairs where cell B would be out of bounds, or where the computed center falls outside
the domain, are discarded.

For pairs whose velocity vectors are parallel (pure translation, no rotation):
the contribution is discarded — pure translation has no instantaneous center.

**ω formula:** `ω = (arm × v) / |arm|²`
where `arm` is the vector from the rotation center to the cell, and `v` is the
velocity at that cell. Dividing by `|arm|²` normalizes out distance — every pair
contributes true angular velocity regardless of how far apart the cells are.

## Architecture

Single `index.html` entry point. JavaScript split across well-named module files.
Both fields run entirely on WebGPU — no WebGL. No build step — native ES modules.

```
index.html
src/
  main.js                       — bootstrap, canvas sizing, render loop
  config/
    SimulationConfig.js         — all named constants; single source of truth
  simulation/
    FluidField.js               — Field A: wraps physics step, exposes velocityTexture
    WebGPUNavierStokesStep.js   — physics step (advection, pressure projection,
                                  vorticity confinement, inject, render) — all WebGPU
    WebGPURotationField.js      — Field B: WebGPU compute pipeline (accumulate → reduce → render)
    PhysicsRegistry.js          — maps physics model identifiers to step implementations
  interaction/
    MouseInjector.js            — translates mouse events into field impulses
    PatternInjector.js          — double-click injection of velocity patterns on Field B
    FocusMaskInteractor.js      — click/drag on Field B to restrict rotation to a focus circle
  ui/
    ControlPanel.js             — all DOM slider creation and wiring
  gpu/
    WebGPUDevice.js             — WebGPU adapter/device acquisition
    UniformWriter.js            — sequential writer for WebGPU uniform buffers
  shaders/
    velocity_advect.wgsl              — bicubic Catmull-Rom semi-Lagrangian advection
    velocity_divergence.wgsl          — ∇·v
    velocity_pressure.wgsl            — 9-point Mehrstellen isotropic pressure Jacobi
    velocity_subtract_gradient.wgsl   — v − ∇p; reflect boundary
    velocity_inject_impulse.wgsl      — Gaussian impulse for mouse strokes
    velocity_inject_disk.wgsl         — single-pass filled disk (spin/explode/implode)
    velocity_vorticity_curl.wgsl      — computes scalar curl field
    velocity_vorticity_confinement.wgsl — applies confinement force to re-energise vortices
    velocity_noise.wgsl               — per-cell pseudo-random velocity
    velocity_render.wgsl              — render pipeline: Reinhard tonemapping + HSV encoding
    rotation_compute.wgsl             — WGSL compute: accumulates ω into K atomic i32 buffers
    rotation_reduce.wgsl              — WGSL compute: sums K buffers → flat f32 output buffer
    rotation_render.wgsl              — WGSL render: output buffer → canvas (orange/blue tones)
```

## Code Quality Rules — Non-Negotiable

This is not a "quick JS demo". The fact that the output is a browser app does not
excuse sloppy code. The following rules apply without exception.

**Git**
- Do NOT run `git commit` (or any destructive git command) without an explicit instruction
  from the user. Make the change, describe what was done, and wait.

**Naming**
- Every function name must describe what it does, not how.
  `stepNavierStokes()` not `update()`.
  `computeInstantaneousRotationCenter()` not `calc()`.
- No single-letter variables outside of shader WGSL math (where `u`, `v` are domain-standard).
- No abbreviations that require context to decode. `viscosity` not `visc`. `velocity` not `vel`.

**Decomposition**
- No function does more than one thing.
- Physics logic does not live inside render loops. Render loops call named physics steps.
- A reader must be able to understand the top-level flow by reading only function names,
  without reading any function bodies.

**No JS idiom rot**
- No callback pyramids. Async where needed, promises otherwise.
- No magic numbers inline. All constants named and grouped at the top of their module.
- No `var`. No implicit globals. Strict mode everywhere.
- No comment that explains *what* the code does (the code does that). Comments explain *why*
  when the why is non-obvious (e.g. a numerical stability trick).

**Shaders**
- WGSL functions are named descriptively.
- No monolithic entry point that does everything. Break into named functions.

**State**
- No shared mutable globals. State lives in class instances passed explicitly.
- Exception: `PHYSICS_DEFAULTS`, `MOUSE_DEFAULTS`, and `ROTATION_FIELD` in
  `SimulationConfig.js` are intentionally mutable — the UI writes to them and the
  simulation reads them live each frame, so slider changes take effect immediately
  without any extra wiring.

## Layout

Both fields are always visible simultaneously, side by side:

```
[ Field A — Velocity ]  [ Field B — Rotation Centers ]
```

No toggle, no split-view switch. Both render every frame.

Both canvases are WebGPU. They sit inside a flex container
with `DISPLAY_GAP` (px) of space between them.

Each canvas size is computed dynamically from `window.innerWidth` and `window.innerHeight`
so both fields always fit on screen at the largest integer multiple of `DISPLAY_SCALE` that fits.

**Field A** — velocity magnitude mapped to brightness via HSV (hue = direction, value = speed).
Brightness uses Reinhard tonemapping: `x / (x + c)` where `c = velocityToneMidpoint`.

**Field B — color encoding:**
- Pure black = zero (no rotation contribution)
- Orange = positive (counter-clockwise rotation)
- Blue = negative (clockwise rotation)
- Brightness encodes magnitude via Reinhard tonemapping with `rotationToneMidpoint`

Color constants (`rotationPositive`, `rotationNegative`, `rotationZero`) live in
`SimulationConfig.js` as named exports, not inline hex values.

## Parameters — All Named, None Hardcoded

Every tuneable value lives in `src/config/SimulationConfig.js` as a named export.
No magic numbers anywhere else in the codebase. Current structure:

```js
export const GRID_SIZE = 256;

export const DISPLAY_SCALE = 7;  // pixels per grid cell (max; clamped to fit screen)
export const DISPLAY_GAP = 32;   // pixel gap between the two fields

// Reference velocity (px/s) at which speedSensitivity=1 gives a 1× strength multiplier.
export const MOUSE_SPEED_REFERENCE = 400;

// Reference grid size for Field B brightness auto-compensation.
export const ROTATION_REFERENCE_GRID_SIZE = 100;

export const PHYSICS_DEFAULTS = {
  damping:             0.9999999, // per-second velocity retention; applied as pow(damping, dt)
  pressureIterations:  40,        // fewer = more compressible / gas-like
  simulationSpeed:     2.0,       // playback speed multiplier (see render loop for semantics)
  boundaryMode:        0,         // 0=wrap 1=absorb 2=reflect
  vorticityStrength:   0.0,       // vorticity confinement ε; 0 = disabled
};

export const MOUSE_DEFAULTS = {
  impulseRadius:    2.0,   // brush radius in grid cells
  impulseStrength:  100.0,
  speedSensitivity: 1.0,   // 0 = constant strength; 1 = fully speed-scaled
  patternScale:     0.7,   // pattern size as fraction of field
};

export const ROTATION_FIELD = {
  parallelThreshold: 0.001,  // below this cross-product magnitude → discard pair
  accumulationScale: 1.0,
  pairDistance:      0.02,   // annulus center as fraction of gridSize/2
  distanceDelta:     0.02,   // annulus half-width as fraction of gridSize/2
  sampleStride:      1,      // skip cells: stride S samples every S-th cell in each axis
  showMask:          3,      // bit0=show CCW(positive) bit1=show CW(negative); 3=both
  maskCenter:        null,   // [col, row] grid coords of focus circle center; null = no mask
  maskRadius:        null,   // focus circle radius in grid cells
};

export const RENDER_DEFAULTS = {
  velocityToneMidpoint: 30.0,
  rotationToneMidpoint: 0.3,
};

export const COLORS = {
  rotationPositive: [1.0, 0.5, 0.0],  // orange — counter-clockwise
  rotationNegative: [0.0, 0.6, 1.0],  // blue   — clockwise
  rotationZero:     [0.0, 0.0, 0.0],  // black  — no contribution
};
```

## UI Controls

`ControlPanel.js` builds all sliders from DOM. Available slider types:
- `_addLogSlider` — log-scaled, center = default value at construction
- `_addLinearSlider` — linear, float display
- `_addIntSlider` — linear, integer display
- `_addSlider` — generic; caller supplies `formatValue` and `onChange` for custom curves
- `_addSymmetricPowerSlider` — power curve symmetric around center; exponent < 1 gives
  finer control near center
- `_addContributionToggles` — two toggle buttons (CCW / CW) that independently set bits
  in `ROTATION_FIELD.showMask`; active = full opacity, inactive = 0.35 opacity

Physics sliders appear below Field A. Brightness sliders appear below their respective field.

Current sliders:
- **Brightness** (under each field) — adjusts tone midpoints
- **Boundary** (dropdown) — Wrap / Absorb / Reflect; Grid size input on same row
- **Brush radius** — `MOUSE_DEFAULTS.impulseRadius` (log, 0.5–256)
- **Brush strength** — `MOUSE_DEFAULTS.impulseStrength` (log, 1–500)
- **Speed sensitivity** — `MOUSE_DEFAULTS.speedSensitivity` (linear, 0–1); 0 = constant
  strength regardless of mouse speed; 1 = fully speed-scaled
- **dt** — `PHYSICS_DEFAULTS.simulationSpeed` (log, 0.1–10); qualitatively changes flow
- **Damping loss** — `1 - damping` (log, near-zero to 0.2)
- **Vorticity** — `vorticityStrength` (power-curve, exponent 2, 0–50); re-energises vortices;
  0 = off; power curve gives finer control near zero
- **Incompressibility (liquid / gas)** — inverted pressure iterations (1–100);
  right = liquid (many iters, incompressible), left = gas (few iters, compressible)
- **Pattern size** (under Field B) — `MOUSE_DEFAULTS.patternScale`
- **Show contribution** (under Field B) — two toggle buttons CCW / CW; independently
  enable/disable positive and negative ω contributions via `ROTATION_FIELD.showMask` bit flags
- **Pair distance** (under Field B) — `ROTATION_FIELD.pairDistance` (log, 0.001–1);
  center of the annulus as fraction of gridSize/2
- **Distance delta** (under Field B) — `ROTATION_FIELD.distanceDelta` (log, 0.001–1);
  half-width of the annulus; pairs included in `[distance − delta, distance + delta] × gridSize/2`
- **Sample stride** (under Field B) — powers of 2 (1, 2, 4, 8, 16, 32); skips cells in both
  axes, reducing computation by stride²; brightness auto-compensated

**Pattern injection** (`PatternInjector.js` — double-click on Field B):
A dropdown below Field B selects the injection pattern. Double-clicking injects at
that UV position in Field A space. Uses current brush radius/strength. Adds to existing
velocity, never resets it. Initial pattern is queued for first render frame (not applied
at startup) to guarantee a stable GPU state.

Current patterns:
- **Disk — spin / explode / implode** — filled disk injected via single shader pass
  (`velocity_inject_disk.wgsl`); Gaussian radial falloff avoids boundary divergence artifacts
- **Polygons** (circle, triangle, square … decagon) — perimeter with CCW tangential velocity
- **Parallel stripes** — alternating ←→ flow (seeds Kelvin-Helmholtz instability)
- **Square grid lines** — crossing orthogonal jets
- **Triangular grid lines** — three families at 0°/60°/120°
- **Scattered points** — hexagonal packing, radially outward
- **Random noise** — per-cell pseudo-random velocity via `velocity_noise.wgsl`

**Mouse injection** (`MouseInjector.js`):
- `mousedown` on Field A canvas activates injection; clicks on Field B are ignored
- Between frames, the full stroke segment (prev → current position) is interpolated
  at `0.75 × brushRadius` step spacing so fast motion leaves no gaps
- Mouse leaving field bounds resets `_previousPosition` to prevent phantom strokes on re-entry
- Impulse strength is speed-scaled: `strength = impulseStrength × (1 − sensitivity + sensitivity × speedFactor)`
  where `speedFactor = pixelsPerSecond / MOUSE_SPEED_REFERENCE`; `speedSensitivity = 0`
  gives constant strength, `= 1` scales fully with mouse velocity

**Focus mask** (`FocusMaskInteractor.js`):
- Click or drag on Field B sets a circular focus mask at that grid position
- While the mask is active, Field B dispatches only the bounding box of the circle
  (O(R²) threads instead of O(N²)); pairs where cell B or the computed center falls
  outside the circle are discarded
- A Canvas 2D dashed circle overlay is drawn over Field A to visualise the active region
- Quick click on an already-active mask clears it; Escape always clears
- Mask center, radius, and all other parameters are URL-persisted via `buildShareUrl()`

## Boundary Conditions

Three modes, selected via dropdown:

**Wrap (0):** Periodic — fields connect edge to edge via `wrapI()` in all shaders.

**Absorb (1):** Open boundary — fluid exits freely.
- Advection: backtracked positions outside domain return `vec2(0)` (no energy smuggled in)
- Pressure: Dirichlet `p=0` at ghost cells (consistent in both solve and gradient steps)
- Velocity: no forced zeroing at boundary cells — fluid exits through natural advection

**Reflect (2):** Solid wall — normal velocity component negated at boundary cells.
- Pressure: Neumann (clamp) at ghost cells
- Gradient subtract: normal component negated at edge cells

Field B respects boundary mode: pair displacement uses the shortest torus path only in
Wrap mode; in other modes the direct vector is used, out-of-bounds cells B are skipped,
and computed rotation centers outside the domain are discarded.

## Advection

Semi-Lagrangian advection with Catmull-Rom bicubic interpolation (`velocity_advect.wgsl`).
Each cell traces back `position − velocity × dt` in cell-centre space, then samples
the velocity field with a 4×4 Catmull-Rom kernel (separable in x and y). The bicubic
interpolation suppresses the axis-aligned anisotropy artifacts that bilinear produces.

Explicit Jacobi diffusion is omitted. The semi-Lagrangian scheme introduces numerical
diffusion of approximately `h²/(2·dt)`, which is sufficient and removes the need for a
separate diffusion pass.

Damping is applied as `pow(PHYSICS_DEFAULTS.damping, deltaTime)` — a per-second
retention factor independent of frame rate.

## Pressure Solver

Nine-point Mehrstellen isotropic Laplacian (`velocity_pressure.wgsl`) instead of the
standard 5-point stencil. The 9-point formula weights cardinal neighbors ×4 and diagonal
neighbors ×1, divides by 20: `(4*(left+right+bottom+top) + (sw+se+nw+ne) − 6·divergence) / 20`.
This reduces the directional bias that causes diagonal artifacts with the 5-point stencil.

Pressure is warm-started between frames (previous frame's solution as initial guess),
improving Jacobi convergence without any extra zeroing cost.

## Vorticity Confinement

Two-pass process after advection, before pressure projection:
1. `velocity_vorticity_curl.wgsl` — computes scalar curl `ω = ∂vy/∂x − ∂vx/∂y`
2. `velocity_vorticity_confinement.wgsl` — computes `∇|ω|`, normalises it to unit vector `η`,
   applies force `ε × ω × (η.y, −η.x) × dt` to velocity

The force re-energises vortex cores that numerical diffusion would otherwise smooth away.
Skipped entirely when `vorticityStrength == 0` (no GPU cost).

## Field A — WebGPU Compute Pipeline

Field A physics runs entirely on WebGPU (`WebGPUNavierStokesStep.js`).
One encoder per `step()` call, submitted once:

1. **Advect** (`velocity_advect.wgsl`) — semi-Lagrangian bicubic; ping-pong velocity textures.
2. **Vorticity curl** (`velocity_vorticity_curl.wgsl`) — scalar curl → `r32float` curl texture.
3. **Vorticity confinement** (`velocity_vorticity_confinement.wgsl`) — adds confinement force;
   skipped if `vorticityStrength == 0`.
4. **Divergence** (`velocity_divergence.wgsl`) — ∇·v → `r32float` divergence texture.
5. **Pressure Jacobi** (`velocity_pressure.wgsl`) — N iterations; ping-pong pressure textures.
6. **Subtract gradient** (`velocity_subtract_gradient.wgsl`) — v − ∇p; enforce incompressibility.

Inject operations (impulse, disk, noise) each use a separate per-call encoder+submit so
multiple injections per frame compose correctly without needing a params buffer array.

**Texture convention:** row 0 = physical bottom (y-up). The render shader maps
`uv.y = 0` (screen bottom) to row 0, preserving orientation. Mouse input flips y:
`v = 1 − pixelY / fieldSize`.

**UniformWriter** (`gpu/UniformWriter.js`) — sequential writer for WebGPU uniform buffers.
Writes u32/f32 fields in declaration order, eliminating manual index arithmetic. Padding
fields must be written explicitly via `pad()` to keep slot count in sync with the WGSL
struct — wrong field count → detectable size mismatch at `writeBuffer` time.

## Field B — WebGPU Compute Pipeline

Field B rotation accumulation runs entirely on WebGPU (`WebGPURotationField.js`).
Reads velocity directly from the GPUTexture exposed by `FluidField.velocityTexture` —
no CPU round-trip. The pipeline has three passes per frame in one encoder:

1. **Compute pass** (`rotation_compute.wgsl`) — N² threads (one per cell i), each
   iterates the precomputed annulus offset table (flat i32 buffer of `[dCol, dRow]` pairs,
   built CPU-side when `pairDistance`/`distanceDelta` change, uploaded once). No per-pair
   distance check in the shader — every entry is guaranteed to lie in the annulus. For each
   valid pair the instantaneous rotation center is computed and ω averaged from both
   cells is accumulated into one of K=16 atomic i32 accumulation buffers selected by
   `workgroupId % K`, reducing CAS contention K-fold. Accumulation uses a CAS-based
   f32 atomic add (bit-cast through i32; `clearBuffer(0)` initialises correctly since
   0x00000000 is 0.0 in IEEE 754). `showMask` bit flags filter contributions by sign
   (bit0=CCW/positive, bit1=CW/negative) before accumulation.

   When the focus mask is active, threads cover only the bounding box of the circle
   (O(R²) instead of O(N²)), and pairs where cell B or the center is outside the
   circle are discarded.

2. **Reduce pass** (`rotation_reduce.wgsl`) — N² threads sum the K accumulation
   buffers into a flat f32 output buffer.

3. **Render pass** (`rotation_render.wgsl`) — a full-screen quad samples the output
   buffer and applies Reinhard tonemapping with orange/blue color encoding.

The compute bind group is rebuilt each frame (takes the current velocity GPUTexture
view) — cheap since `getBindGroupLayout` is not a GPU operation.

**Output cache:** when the simulation is paused and only render params changed (e.g.
brightness slider), `recomputeFrom()` skips compute+reduce and re-runs only the render
pass from the cached `_outputBuffer`. Detection uses `FluidField.stepGeneration` (an
integer incremented on every `step()` call) to identify velocity changes reliably —
texture reference identity is not used because the ping-pong swap count per step can
be even, causing the same texture object to appear on successive frames.

## Extensibility Requirements

The code must be written so the following changes require minimal surgery:

**Adding a new physics model** (e.g. wave equation, reaction-diffusion):
- Implement a new class with the same interface as `WebGPUNavierStokesStep.js`:
  - `step(deltaTime)`
  - `injectImpulse(position, direction, radius, strength)`
  - `injectDisk(center, radiusUv, strength, mode)`
  - `injectNoise(strength, seed)`
  - `clearVelocity()`
  - `renderToCanvas()`
  - `get velocityTexture`  — current GPUTexture (rg32float)
  - `dispose()`
- Register it in `PhysicsRegistry.js`
- The rest of the system does not change

**Changing grid size:**
- Change `GRID_SIZE` in `SimulationConfig.js`
- Nothing else changes

**Adding UI controls:**
- `SimulationConfig.js` values are the single source of truth
- UI reads from and writes to config only, never touches simulation internals directly
- Add sliders via `ControlPanel._addLogSlider()`, `_addLinearSlider()`, `_addIntSlider()`,
  `_addSlider()`, or `_addSymmetricPowerSlider()`
- Every new control **must** have a corresponding URL parameter: add it to `DEFAULTS`,
  parse it with `_float`/`_int`/`_str` in `PHYSICS_DEFAULTS` / `MOUSE_DEFAULTS` / etc.,
  and serialize it in `buildShareUrl()`. This is required — share URLs must fully
  reproduce the simulation state.

## Pair Interaction Range

`ROTATION_FIELD.pairDistance` and `ROTATION_FIELD.distanceDelta` together define an
annulus of pair distances that contribute to Field B. Both are fractions of `gridSize/2`.

- **pairDistance** — center of the annulus (log slider, 0.001–1).
- **distanceDelta** — half-width of the annulus (log slider, 0.001–1).
- Pairs are included when their distance falls in `[pairDistance − distanceDelta, pairDistance + distanceDelta] × gridSize/2`.
- `distanceDelta ≥ pairDistance` makes the inner radius clamp to 0 (pure disk, no hole).

The annulus is precomputed CPU-side (`_rebuildAnnulusOffsets`) whenever either parameter
changes and uploaded as a flat `i32` storage buffer (`[dCol0, dRow0, dCol1, dRow1, ...]`).
The shader iterates this list directly — no distance check per pair, no bounding-box scan.
The buffer is sized at `gridSize² × 8` bytes (safe upper bound; worst case ≈ 51k entries
for a full inscribed disk at `gridSize=256`).

**Sample stride** reduces computation further: stride S samples every S-th cell in each
axis, reducing contributing pairs by S². Field B brightness is auto-compensated by
multiplying `accumulationScale` by `sampleStride²` and by `gridSize / ROTATION_REFERENCE_GRID_SIZE`
so the display stays calibrated across both parameters.

## Known Limitations

**No explicit viscosity slider.** The semi-Lagrangian advection scheme introduces numerical
diffusion. The `dt` slider (`simulationSpeed`) qualitatively changes the flow regime and
is the most meaningful control for flow character.

**`simulationSpeed` is not a pure time-scale.** At `speed ≤ 1` the render loop runs one
physics step per frame with `dt = PHYSICS_DT × speed` (smooth, slight regime shift). At
`speed > 1` it runs `round(speed)` sub-steps per frame at fixed `PHYSICS_DT = 1/60`
(correct regime, multiple steps). True time-scaling is not achievable with Jacobi pressure:
the Jacobi ringing frequency is set by the grid and stencil eigenvalues, not by `dt` — so
abrupt `dt` changes produce transient oscillations at a fixed grid-dependent frequency
regardless of the magnitude of the change. A CG or multigrid solver would eliminate this,
at significant GPU cost (global reductions per iteration).

**Field B GPU load scales with N².** At gridSize=256 the compute shader dispatches 65536
threads, each iterating the precomputed annulus offset list. Use `pairDistance`/`distanceDelta`
to control annulus size and `sampleStride` to sub-sample the grid. The focus mask further
reduces dispatch to O(R²) for the selected region.

**CFL instability at very high impulse strength.** Semi-Lagrangian advection requires
`velocity × dt / gridSize < 1` per cell. At extreme strength values the condition is
violated and the field becomes chaotic. This is a fundamental limit of the scheme.

## Non-goals

- No UI framework. Plain HTML controls.
- No build step. Native ES modules (`type="module"`).
- No TypeScript (but code should read as if it were typed).

## TODOs
- export animation
- reaction-diffusion
- пресеты
- temperature field / buoyancy (Boussinesq)
- BFECC advection for lower numerical diffusion

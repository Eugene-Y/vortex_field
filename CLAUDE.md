# Vortex Field — Project Brief

## Concept

Two coupled WebGL fields on a 50×50 grid (configurable via `GRID_SIZE`).

**Field A — Velocity field.**
A 2D fluid simulation (Navier-Stokes) on a grid. The user interacts with the mouse,
injecting impulses into the field. Energy propagates, diffuses, and decays according
to configurable parameters (damping, brush radius, brush strength).

**Field B — Instantaneous rotation field.**
Reset to zero every frame. For every pair of cells (i, j) in Field A, compute the
instantaneous center of rotation of those two cells relative to each other. The
signed magnitude of that relative rotation is accumulated into the cell of Field B
whose grid coordinate corresponds to that center of rotation. Sign encodes direction
(clockwise vs counter-clockwise). Magnitude encodes strength.

This is not an only-local neighborhood approximation. By default, all unique pairs in the radius of GRID_SIZE/2 are evaluated. Since the instantaneous center of rotation and angular velocity are symmetric for any pair (A, B) and (B, A), the GPU kernel only computes unique pairs (`indexA < indexB`) and doubles their contribution, halving the required math and blending workload. The interaction can be further restricted spatially via the `pairRange` parameter.

For pairs whose velocity vectors are parallel (pure translation, no rotation):
the contribution is discarded — pure translation has no instantaneous center.

**ω formula:** `ω = (arm × vA) / |arm|²`
where `arm` is the vector from cell j to cell i, and `vA` is the velocity at cell i.
Dividing by `|arm|²` normalizes out distance — every pair contributes true angular
velocity regardless of how far apart the cells are.

## Architecture

Single `index.html` entry point. JavaScript split across well-named module files.
WebGL2 for all field computation and rendering. No build step — native ES modules.

```
index.html
src/
  main.js                  — bootstrap, canvas sizing, render loop
  config/
    SimulationConfig.js    — all named constants; single source of truth
  simulation/
    FluidField.js          — Field A: wraps physics step, exposes velocityTexture
    RotationField.js       — Field B: per-frame rotation accumulation via GPU kernel
    NavierStokesStep.js    — physics step (advection, diffusion, pressure projection)
    PairRotationKernel.js  — draws GRID_SIZE⁴ points, one per pair, additive blending
  rendering/
    FieldRenderer.js       — renders either field to the appropriate viewport
  interaction/
    MouseInjector.js       — translates mouse events into field impulses
  ui/
    ControlPanel.js        — all DOM slider creation and wiring
  gl/
    GlContext.js           — WebGL2 context setup; requires EXT_color_buffer_float
                             and EXT_float_blend (critical — see below)
    Framebuffer.js         — PingPongFramebuffer and SingleFramebuffer abstractions
    ShaderProgram.js       — shader compilation, uniform cache, bind/set methods
  shaders/
    common.vert            — full-screen quad vertex shader (shared)
    advect.frag
    diffuse.frag
    divergence.frag
    pressure.frag
    subtract_gradient.frag
    inject_impulse.frag
    rotation_accumulate.vert  — per-pair vertex shader: computes center, clips to grid
    rotation_accumulate.frag  — outputs ω contribution for additive blending
    render.frag               — Reinhard tonemapping for both fields; HSV for velocity,
                                orange/blue for rotation
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
- No single-letter variables outside of shader GLSL math (where `u`, `v` are domain-standard).
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
- GLSL functions are also named descriptively.
- No monolithic `main()` that does everything. Break into named functions.

**State**
- No shared mutable globals. State lives in class instances passed explicitly.
- Exception: `PHYSICS_DEFAULTS` and `MOUSE_DEFAULTS` in `SimulationConfig.js` are
  intentionally mutable — the UI writes to them and the physics step reads them live
  each frame, so slider changes take effect immediately without any extra wiring.

## Layout

Both fields are always visible simultaneously, side by side:

```
[ Field A — Velocity ]  [ Field B — Rotation Centers ]
```

No toggle, no split-view switch. Both render every frame.

Single canvas, single GL context. The canvas width is `CANVAS_FIELD_SIZE * 2 + DISPLAY_GAP`.
Two viewports are set per frame — one for each field. `DISPLAY_GAP` (px) separates them.

`CANVAS_FIELD_SIZE` is computed dynamically from `window.innerWidth` and `window.innerHeight`
so both fields always fit on screen at the largest integer scale that fits.

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
export const GRID_SIZE = 50;

export const DISPLAY_SCALE = 8;  // pixels per grid cell (max; clamped to fit screen)
export const DISPLAY_GAP = 32;   // pixel gap between the two fields

export const PHYSICS_DEFAULTS = {
  viscosity: 0.001,          // explicit viscosity (NOTE: see Known Limitations)
  damping: 0.995,            // per-frame velocity retention (1 = no loss)
  diffusionIterations: 20,
  pressureIterations: 40,
};

export const MOUSE_DEFAULTS = {
  impulseRadius: 2.0,        // brush radius in grid cells
  impulseStrength: 100.0,
};

export const ROTATION_FIELD = {
  parallelThreshold: 0.001,  // below this cross-product magnitude → discard pair
  accumulationScale: 1.0,
  pairRange: 1.0,             // interaction radius as fraction of gridSize/2 (see below)
};

export const RENDER_DEFAULTS = {
  velocityToneMidpoint: 30.0,   // velocity magnitude that maps to 50% brightness
  rotationToneMidpoint: 0.3,    // rotation ω that maps to 50% brightness
};

export const COLORS = {
  rotationPositive: [1.0, 0.5, 0.0],  // orange — counter-clockwise
  rotationNegative: [0.0, 0.6, 1.0],  // blue   — clockwise
  rotationZero:     [0.0, 0.0, 0.0],  // black  — no contribution
};
```

## UI Controls

`ControlPanel.js` builds all sliders from DOM. All sliders are log-scaled (100 integer steps,
linear in log space between min and max). The center position corresponds to the default value
at construction time.

Current sliders:
- **Brightness** (under each field) — adjusts `velocityToneMidpoint` / `rotationToneMidpoint`
- **Brush radius** — adjusts `MOUSE_DEFAULTS.impulseRadius` (range 0.5–20)
- **Brush strength** — adjusts `MOUSE_DEFAULTS.impulseStrength` (range 5–500)
- **Damping loss** — adjusts `PHYSICS_DEFAULTS.damping` via the transform
  `loss = 1 - damping` (range 0.0005–0.2), displayed in loss space for clean log scale

Physics sliders appear below Field A. Brightness sliders appear below their respective field.

**Pattern injection** (`PatternInjector.js` — double-click on Field B):
A dropdown below Field B selects the injection pattern. Double-clicking anywhere on Field B
injects it at the clicked grid position (same UV coordinates as Field A). Uses current brush
radius and strength from the sliders. Adds to existing velocity, never resets it.

Current patterns:
- **Polygons** (circle, triangle, square, pentagon, hexagon, heptagon, octagon, nonagon,
  decagon) — 32 points along the perimeter with CCW tangential velocity
- **Parallel stripes** — 5 horizontal stripes × 8 points, alternating ←→ flow
  (seeds Kelvin-Helmholtz shear instability)
- **Square grid lines** — 5 horizontal + 5 vertical lines of 7 points each, orthogonal
  flow directions (↔ and ↕), creates crossing jets and junction vortices
- **Triangular grid lines** — 3 families of 5 parallel lines at 0°/60°/120°, each flowing
  along its line direction, creates hexagonal interference
- **Scattered points** — hexagonal packing (center + ring of 6 + ring of 12), each pointing
  radially outward from the pattern center
- **Random noise** — per-cell pseudo-random velocity added via `noise.frag` shader; uses
  current brush strength. Uniform coverage, no position dependency.
- **Reset** — clears both ping-pong velocity buffers to zero

## WebGL Requirements

`GlContext.js` requires two extensions at startup (throws if unavailable):
- `EXT_color_buffer_float` — needed to render into R32F/RG32F framebuffers
- `EXT_float_blend` — needed for additive blending into float framebuffers (Field B)

Without `EXT_float_blend`, additive blending into a float texture is undefined behavior:
each draw call would overwrite rather than accumulate, so Field B would show only one
random pair per pixel instead of the correct sum over all pairs.

Physics framebuffers use `TEXTURE_WRAP_S/T = REPEAT` (periodic boundary conditions).
This avoids wall accumulation artifacts that appear with `CLAMP_TO_EDGE`.

## Extensibility Requirements

The code must be written so the following changes require minimal surgery:

**Adding a new physics model** (e.g. wave equation, reaction-diffusion):
- Implement a new class with the same interface as `NavierStokesStep.js`:
  - `step(deltaTime, quadVao)`
  - `injectImpulse(position, direction, radius, strength, quadVao)`
  - `get velocityTexture`
  - `dispose()`
- Register it in `PhysicsRegistry.js`
- The rest of the system does not change

**Changing grid size:**
- Change `GRID_SIZE` in `SimulationConfig.js`
- Nothing else changes

**Adding UI controls:**
- `SimulationConfig.js` values are the single source of truth
- UI reads from and writes to config only, never touches simulation internals directly
- Add sliders via `ControlPanel._addLogSlider()` or `ControlPanel._addSlider()`
- Every new control **must** have a corresponding URL parameter: add it to `DEFAULTS`,
  parse it with `_float`/`_int`/`_str` in `PHYSICS_DEFAULTS` / `MOUSE_DEFAULTS` / etc.,
  and serialize it in `buildShareUrl()`. This is required — share URLs must fully
  reproduce the simulation state.

**Adding a new color map:**
- Add a named function to a `ColorMap.js` module
- Pass it as a parameter to `FieldRenderer`

## Pair Interaction Range

`ROTATION_FIELD.pairRange` (slider "Pair range" under Field B, range 0–1) limits which
pairs contribute to Field B. For a pair (A, B), B is included only if its periodic
minimum-image distance from A satisfies:

```
|d_AB| ≤ pairRange × gridSize / 2
```

Equivalently, the interaction **diameter** is `pairRange × gridSize`.

- At **1.0** all pairs within the inscribed circle of the periodic grid contribute.
  Corner pairs whose distance exceeds `gridSize/2` are excluded at all range values
  (they are the diagonal extremes of the periodic torus and represent ≈22% of all pairs).
- At **0.5** only pairs within `gridSize/4` radius contribute — a quarter of the field.
- At **→ 0** the field goes dark (no contributing pairs).

The discard is a single early-return in the vertex shader (`rotation_accumulate.vert`).
The draw call still submits `GRID_SIZE⁴` vertices, so GPU work does not decrease with
smaller range — only fragment writes and blending are skipped. A future optimisation
could reduce the draw count for very small ranges.

## Known Limitations

**Viscosity slider was removed.** The semi-Lagrangian advection scheme introduces numerical
diffusion of approximately `h²/(2·dt)` ≈ 0.013 (at `GRID_SIZE=50`, `dt≈0.016`). Any explicit
viscosity value below this threshold has no visible effect — the numerical diffusion dominates.
A viscosity slider is therefore misleading and was removed. If viscosity control becomes
important, it would require switching to a scheme with lower numerical diffusion (e.g. BFECC
or MacCormack advection).

## Non-goals

- No UI framework. Plain HTML controls.
- No build step. Native ES modules (`type="module"`).
- No TypeScript (but code should read as if it were typed).

## TODOs
- export animation
- reaction-diffusion
- пресеты

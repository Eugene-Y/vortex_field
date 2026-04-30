# Vortex Field — Project Brief

## Concept

Two coupled WebGL fields on a configurable grid (default 64×64, set via `GRID_SIZE`).

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
    NavierStokesStep.js    — physics step (advection, diffusion, pressure projection,
                             vorticity confinement)
    PairRotationKernel.js  — draws GRID_SIZE⁴ points, one per pair, additive blending
  rendering/
    FieldRenderer.js       — renders either field to the appropriate viewport
  interaction/
    MouseInjector.js       — translates mouse events into field impulses
    PatternInjector.js     — double-click injection of velocity patterns on Field B
  ui/
    ControlPanel.js        — all DOM slider creation and wiring
  gl/
    GlContext.js           — WebGL2 context setup; requires EXT_color_buffer_float
                             and EXT_float_blend (critical — see below)
    Framebuffer.js         — PingPongFramebuffer and SingleFramebuffer abstractions
    ShaderProgram.js       — shader compilation, uniform cache, bind/set methods
  shaders/
    common.vert                — full-screen quad vertex shader (shared)
    advect.frag
    diffuse.frag
    divergence.frag
    pressure.frag
    subtract_gradient.frag
    inject_impulse.frag        — Gaussian impulse for mouse strokes
    inject_disk.frag           — single-pass filled disk injection (spin/explode/implode)
    vorticity_curl.frag        — computes scalar curl field
    vorticity_confinement.frag — applies confinement force to re-energise vortices
    noise.frag
    rotation_accumulate.vert   — per-pair vertex shader: computes center, clips to grid
    rotation_accumulate.frag   — outputs ω contribution for additive blending
    render.frag                — Reinhard tonemapping for both fields; HSV for velocity,
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
export const GRID_SIZE = 64;

export const DISPLAY_SCALE = 8;  // pixels per grid cell (max; clamped to fit screen)
export const DISPLAY_GAP = 32;   // pixel gap between the two fields

export const PHYSICS_DEFAULTS = {
  viscosity:           0.001,    // dominated by numerical diffusion; see Known Limitations
  damping:             0.995,    // per-frame velocity retention (1 = no loss)
  diffusionIterations: 20,
  pressureIterations:  40,       // fewer = more compressible / gas-like
  simulationSpeed:     1.0,      // dt multiplier; qualitatively changes flow character
  boundaryMode:        0,        // 0=wrap 1=absorb 2=reflect
  vorticityStrength:   0.0,      // vorticity confinement ε; 0 = disabled
};

export const MOUSE_DEFAULTS = {
  impulseRadius:   2.0,  // brush radius in grid cells
  impulseStrength: 100.0,
  patternScale:    0.5,  // pattern size as fraction of field
};

export const ROTATION_FIELD = {
  parallelThreshold: 0.001,  // below this cross-product magnitude → discard pair
  accumulationScale: 1.0,
  pairRange:         1.0,    // signed: positive = local-first, negative = distant-first
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
- `_addSymmetricPowerSlider` — power curve symmetric around center; exponent < 1 gives
  finer control near center (used for Pair range)

Physics sliders appear below Field A. Brightness sliders appear below their respective field.

Current sliders:
- **Brightness** (under each field) — adjusts tone midpoints
- **Boundary** (dropdown) — Wrap / Absorb / Reflect; Grid size input on same row
- **Brush radius** — `MOUSE_DEFAULTS.impulseRadius` (log, 0.5–20)
- **Brush strength** — `MOUSE_DEFAULTS.impulseStrength` (log, 1–500)
- **dt** — `PHYSICS_DEFAULTS.simulationSpeed` (log, 0.1–10); qualitatively changes flow
- **Damping loss** — `1 - damping` (log, near-zero to 0.2)
- **Vorticity** — `vorticityStrength` (linear, 0–2); re-energises vortices; 0 = off
- **Incompressibility (liquid / gas)** — inverted pressure iterations (1–100);
  right = liquid (many iters, incompressible), left = gas (few iters, compressible)
- **Pattern size** (under Field B) — `MOUSE_DEFAULTS.patternScale`
- **Pair range** (under Field B) — power-curve symmetric slider, exponent 0.4;
  positive = local pairs first, negative = distant pairs first

**Pattern injection** (`PatternInjector.js` — double-click on Field B):
A dropdown below Field B selects the injection pattern. Double-clicking injects at
that UV position in Field A space. Uses current brush radius/strength. Adds to existing
velocity, never resets it. Initial pattern is queued for first render frame (not applied
at startup) to guarantee stable GL state.

Current patterns:
- **Disk — spin / explode / implode** — filled disk injected via single shader pass
  (`inject_disk.frag`); Gaussian radial falloff avoids boundary divergence artifacts
- **Polygons** (circle, triangle, square … decagon) — perimeter with CCW tangential velocity
- **Parallel stripes** — alternating ←→ flow (seeds Kelvin-Helmholtz instability)
- **Square grid lines** — crossing orthogonal jets
- **Triangular grid lines** — three families at 0°/60°/120°
- **Scattered points** — hexagonal packing, radially outward
- **Random noise** — per-cell pseudo-random velocity via `noise.frag`

**Mouse injection** (`MouseInjector.js`):
- `mousedown` on canvas sets `_pressedInField = true`; injection only fires if drag
  originated on Field A — prevents slider interaction from leaking into the field
- Between frames, the full stroke segment (prev → current position) is interpolated
  at `0.75 × brushRadius` step spacing so fast motion leaves no gaps
- Mouse leaving field bounds resets `_previousPosition` to prevent phantom strokes on re-entry

## Boundary Conditions

Three modes, selected via dropdown:

**Wrap (0):** Periodic — texture wrap `REPEAT`, fields connect edge to edge.

**Absorb (1):** Open boundary — fluid exits freely.
- Advection: backtracked positions outside domain return `vec2(0)` (no energy smuggled in)
- Pressure: Dirichlet `p=0` at ghost cells (consistent in both solve and gradient steps)
- Velocity: no forced zeroing at boundary cells — fluid exits through natural advection
- Diffusion: Neumann (clamp) — velocity continuous across boundary

**Reflect (2):** Solid wall — normal velocity component negated at boundary cells.
- Pressure: Neumann (clamp) at ghost cells
- Diffusion: Neumann (clamp)

## Vorticity Confinement

Two-pass process after advection and diffusion, before pressure projection:
1. `vorticity_curl.frag` — computes scalar curl `ω = ∂vy/∂x − ∂vx/∂y` (raw finite
   differences, not divided by texelSize)
2. `vorticity_confinement.frag` — computes `∇|ω|`, normalises it to unit vector `η`,
   applies force `ε × ω × (η.y, −η.x) × dt` to velocity

The force re-energises vortex cores that numerical diffusion would otherwise smooth away.
Skipped entirely when `vorticityStrength == 0` (no GPU cost).

## WebGL Requirements

`GlContext.js` requires two extensions at startup (throws if unavailable):
- `EXT_color_buffer_float` — needed to render into R32F/RG32F framebuffers
- `EXT_float_blend` — needed for additive blending into float framebuffers (Field B)

Without `EXT_float_blend`, additive blending into a float texture is undefined behavior:
each draw call would overwrite rather than accumulate, so Field B would show only one
random pair per pixel instead of the correct sum over all pairs.

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
- Add sliders via `ControlPanel._addLogSlider()`, `_addLinearSlider()`, `_addIntSlider()`,
  or `_addSymmetricPowerSlider()`
- Every new control **must** have a corresponding URL parameter: add it to `DEFAULTS`,
  parse it with `_float`/`_int`/`_str` in `PHYSICS_DEFAULTS` / `MOUSE_DEFAULTS` / etc.,
  and serialize it in `buildShareUrl()`. This is required — share URLs must fully
  reproduce the simulation state.

**Adding a new color map:**
- Add a named function to a `ColorMap.js` module
- Pass it as a parameter to `FieldRenderer`

## Pair Interaction Range

`ROTATION_FIELD.pairRange` (slider "Pair range" under Field B, range −1 to +1) limits
which pairs contribute to Field B.

- **Positive** values include pairs within `pairRange × gridSize / 2` distance — local first.
- **Negative** values include pairs BEYOND `|pairRange| × gridSize / 2` — distant first.
- At **0** the field goes dark (no contributing pairs).
- At **±1** all pairs within the inscribed circle contribute (corner pairs excluded at all
  values — they exceed `gridSize/2` on the periodic torus, ≈22% of all pairs).

The slider uses a symmetric power curve (exponent 0.4) for finer control near zero.
The discard is a single early-return in the vertex shader. The draw call still submits
`GRID_SIZE⁴` vertices regardless of range — GPU vertex work is constant, only fragment
writes are skipped. Higher |pairRange| = more GPU blend load.

## Known Limitations

**Viscosity slider was removed.** The semi-Lagrangian advection scheme introduces numerical
diffusion of approximately `h²/(2·dt)`. Any explicit viscosity below this threshold has no
visible effect. The `dt` slider (`simulationSpeed`) qualitatively changes the flow regime
and is a more meaningful control.

**Field B disappears at large grid sizes.** Field B draws `GRID_SIZE⁴` points. At
GRID_SIZE=128 that is ~268M points; at 256 it is ~4.3 billion — GPU timeout. Use
`pairRange` to reduce load at larger grid sizes.

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
- интенсивность кисти не точечная а зависит от скорости мыши
- пресеты
- temperature field / buoyancy (Boussinesq)
- BFECC advection for lower numerical diffusion

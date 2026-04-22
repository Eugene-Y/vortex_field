# Vortex Field — Project Brief

## Concept

Two coupled WebGL fields on a 64×64 grid.

**Field A — Velocity field.**
A 2D fluid simulation (Navier-Stokes) on a grid. The user interacts with the mouse,
injecting impulses into the field. Energy propagates, diffuses, and decays according
to configurable parameters (viscosity, damping).

**Field B — Instantaneous rotation field.**
Reset to zero every frame. For every pair of cells (i, j) in Field A, compute the
instantaneous center of rotation of those two cells relative to each other. The
signed magnitude of that relative rotation is accumulated into the cell of Field B
whose grid coordinate corresponds to that center of rotation. Sign encodes direction
(clockwise vs counter-clockwise). Magnitude encodes strength.

This is NOT a local neighborhood approximation. All N² pairs are computed.
Field size is 64×64 = 4096 cells → ~8 million pairs per frame. This is acceptable.

For pairs whose velocity vectors are parallel (pure translation, no rotation):
the contribution is discarded — pure translation has no instantaneous center.

## Architecture

Single `index.html` entry point. JavaScript split across well-named module files.
WebGL for all field computation and rendering.

```
index.html
src/
  main.js                  — bootstrap, event wiring
  simulation/
    FluidField.js          — Field A: Navier-Stokes state and step
    RotationField.js       — Field B: per-frame rotation accumulation
    NavierStokesStep.js    — pure physics step (advection, diffusion, pressure)
    PairRotationKernel.js  — computes instantaneous centers for all pairs
  rendering/
    FieldRenderer.js       — draws either field to screen
    ColorMap.js            — maps scalar/vector values to color
  interaction/
    MouseInjector.js       — translates mouse events into field impulses
  gl/
    GlContext.js           — WebGL context setup and management
    Framebuffer.js         — ping-pong framebuffer abstraction
    ShaderProgram.js       — shader compilation and uniform management
  shaders/
    navier_stokes.frag
    rotation_accumulate.frag
    render.frag
    common.vert
```

## Code Quality Rules — Non-Negotiable

This is not a "quick JS demo". The fact that the output is a browser app does not
excuse sloppy code. The following rules apply without exception.

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

## Layout

Both fields are always visible simultaneously, side by side:

```
[ Field A — Velocity ]  [ Field B — Rotation Centers ]
```

No toggle, no split-view switch. Both render every frame.

**Field A** — velocity magnitude visualized as brightness (or hue-mapped).

**Field B — color encoding:**
- Pure black = zero (no rotation contribution)
- Orange = positive (counter-clockwise rotation)
- Blue = negative (clockwise rotation)
- Brightness encodes magnitude

The color mapping lives in `ColorMap.js` as a named function `mapRotationToColor()`.
The orange and blue hues are named constants, not inline hex values.

## Parameters — All Named, None Hardcoded

Every tuneable value lives in `src/config/SimulationConfig.js` as a named export.
No magic numbers anywhere else in the codebase. Example structure:

```js
export const GRID_SIZE = 64;

export const PHYSICS_DEFAULTS = {
  viscosity: 0.1,
  damping: 0.995,
  diffusionIterations: 20,
};

export const MOUSE_DEFAULTS = {
  impulseRadius: 5,
  impulseStrength: 200,
};

export const ROTATION_FIELD = {
  parallelThreshold: 0.001,  // below this cross-product magnitude → discard pair
  accumulationScale: 1.0,
};

export const COLORS = {
  rotationPositive: [1.0, 0.5, 0.0],  // orange
  rotationNegative: [0.0, 0.6, 1.0],  // blue
  rotationZero:     [0.0, 0.0, 0.0],  // black
};
```

## Extensibility Requirements

The code must be written so the following changes require minimal surgery:

**Adding a new physics model** (e.g. wave equation, reaction-diffusion):
- Implement a new class with the same interface as `NavierStokesStep.js`
- Register it in `PhysicsRegistry.js`
- The rest of the system does not change

**Changing grid size:**
- Change `GRID_SIZE` in `SimulationConfig.js`
- Nothing else changes

**Adding UI controls** (sliders for viscosity, damping, etc.):
- `SimulationConfig.js` values are the single source of truth
- UI reads from and writes to config only, never touches simulation internals directly
- A `ControlPanel.js` module handles all DOM interaction

**Adding a new color map:**
- Add a named function to `ColorMap.js`
- Pass it as a parameter to `FieldRenderer`

## Non-goals for v1

- No UI framework. Plain HTML controls are fine.
- No build step. Native ES modules (`type="module"`).
- No TypeScript (but code should read as if it were typed).
- No sliders yet — but the architecture must make adding them trivial.
'use strict';

import { MOUSE_DEFAULTS, GRID_SIZE, PATTERN_DEFAULTS } from '../config/SimulationConfig.js';

// Consecutive injection points are spaced this fraction of the brush radius apart.
// At 0.75 each Gaussian blob significantly overlaps its neighbours — no visible gaps.
const INJECTION_STEP_FRACTION = 0.75;

const PATTERNS = [
  { value: 'disk-spin',    label: 'Disk — spin'        },
  { value: 'disk-explode', label: 'Disk — explode'     },
  { value: 'disk-implode', label: 'Disk — implode'     },
  { value: 'cross-spin',    label: 'Cross — spin'      },
  { value: 'cross-explode', label: 'Cross — explode'   },
  { value: 'cross-implode', label: 'Cross — implode'   },
  { value: 'circle',    label: 'Circle'                },
  { value: 'triangle',  label: 'Triangle'              },
  { value: 'square',    label: 'Square'                },
  { value: 'pentagon',  label: 'Pentagon'              },
  { value: 'hexagon',   label: 'Hexagon'               },
  { value: 'heptagon',  label: 'Heptagon'              },
  { value: 'octagon',   label: 'Octagon'               },
  { value: 'nonagon',   label: 'Nonagon'               },
  { value: 'decagon',   label: 'Decagon'               },
  { value: 'stripes',   label: 'Parallel stripes'      },
  { value: 'gridlines', label: 'Square grid lines'     },
  { value: 'trilines',  label: 'Triangular grid lines' },
  { value: 'points',    label: 'Scattered points'      },
  { value: 'noise',     label: 'Random noise'          },
];

const POLYGON_SIDES = {
  circle:   0,
  triangle: 3,
  square:   4,
  pentagon: 5,
  hexagon:  6,
  heptagon: 7,
  octagon:  8,
  nonagon:  9,
  decagon:  10,
};

// Returns the number of injection points needed to cover a UV-space path of the
// given length with no gaps, given a brush radius expressed in grid cells.
// The step size equals INJECTION_STEP_FRACTION × radius so consecutive Gaussians overlap.
function stepsForUvLength(uvLength, radiusInCells) {
  const stepSizeInUv = (radiusInCells * INJECTION_STEP_FRACTION) / GRID_SIZE;
  return Math.max(2, Math.ceil(uvLength / stepSizeInUv));
}

export class PatternInjector {
  constructor(canvas, fluidField, fieldSize, displayGap, dropdownContainer) {
    this._canvas     = canvas;
    this._fluidField = fluidField;
    this._fieldSize  = fieldSize;
    this._displayGap = displayGap;
    this._pending    = null;
    this._select     = this._buildDropdown(dropdownContainer);

    this._onDblClick = this._onDblClick.bind(this);
    canvas.addEventListener('dblclick', this._onDblClick);
  }

  injectAt(center, quadVao) {
    this._execute(this._select.value, center, quadVao);
  }

  // Queues an injection to be applied on the next renderFrame call.
  // Use this for startup injection to guarantee a stable GL state.
  queueInitialInjection(center) {
    this._pending = { action: this._select.value, center };
  }

  applyPendingPattern(quadVao) {
    if (!this._pending) return;
    const { action, center } = this._pending;
    this._pending = null;
    this._execute(action, center, quadVao);
  }

  dispose() {
    this._canvas.removeEventListener('dblclick', this._onDblClick);
  }

  _buildDropdown(container) {
    const wrapper = document.createElement('div');
    wrapper.className = 'control-row';

    const label = document.createElement('span');
    label.className = 'control-label';
    label.textContent = 'Inject';

    const select = document.createElement('select');
    select.className = 'pattern-select';
    for (const { value, label: text } of PATTERNS) {
      const option = document.createElement('option');
      option.value = value;
      option.textContent = text;
      select.appendChild(option);
    }
    select.value = PATTERN_DEFAULTS.pattern;

    select.addEventListener('input', () => { PATTERN_DEFAULTS.pattern = select.value; });

    wrapper.appendChild(label);
    wrapper.appendChild(select);
    container.appendChild(wrapper);
    return select;
  }

  _onDblClick(event) {
    const action = this._select.value;
    const center = (action === 'reset' || action === 'noise')
      ? null
      : this._rightFieldClickToUv(event);

    if (action === 'reset' || action === 'noise' || center !== null) {
      this._pending = { action, center };
    }
  }

  // Maps a double-click on the right (rotation) field half to UV [0,1]² in Field A space.
  _rightFieldClickToUv(event) {
    const rect   = this._canvas.getBoundingClientRect();
    const scaleX = this._canvas.width  / rect.width;
    const scaleY = this._canvas.height / rect.height;

    const pixelX = (event.clientX - rect.left) * scaleX;
    const pixelY = (event.clientY - rect.top)  * scaleY;

    const rightStart = this._fieldSize + this._displayGap;
    if (pixelX < rightStart) return null;

    const u = (pixelX - rightStart) / this._fieldSize;
    const v = 1.0 - pixelY / this._fieldSize;

    if (u < 0 || u > 1 || v < 0 || v > 1) return null;
    return [u, v];
  }

  _execute(action, center, quadVao) {
    if (action === 'noise') {
      this._injectNoise(quadVao);
      return;
    }

    const sides = POLYGON_SIDES[action];
    if (sides !== undefined) {
      this._injectPolygon(center, sides, quadVao);
      return;
    }
    if (action === 'disk-spin')    this._injectFilledDisk(center, 'spin', quadVao);
    if (action === 'disk-explode') this._injectFilledDisk(center, 'explode', quadVao);
    if (action === 'disk-implode') this._injectFilledDisk(center, 'implode', quadVao);
    if (action === 'cross-spin')    this._injectCross(center, 'spin', quadVao);
    if (action === 'cross-explode') this._injectCross(center, 'explode', quadVao);
    if (action === 'cross-implode') this._injectCross(center, 'implode', quadVao);
    if (action === 'stripes')   this._injectStripes(center, quadVao);
    if (action === 'gridlines') this._injectSquareGridLines(center, quadVao);
    if (action === 'trilines')  this._injectTriangularGridLines(center, quadVao);
    if (action === 'points')    this._injectScatteredPoints(center, quadVao);
  }

  // Dispatches to circle or regular polygon injection.
  _injectPolygon(center, sides, quadVao) {
    const radius   = MOUSE_DEFAULTS.impulseRadius;
    const strength = MOUSE_DEFAULTS.impulseStrength;
    const patternRadius = MOUSE_DEFAULTS.patternScale * 0.5;

    if (sides < 3) {
      this._injectCircle(center, patternRadius, radius, strength, quadVao);
    } else {
      this._injectRegularPolygon(center, sides, patternRadius, radius, strength, quadVao);
    }
  }

  // Samples the circumference with a step count derived from GRID_SIZE and brush radius
  // so that every grid cell on the ring receives the impulse regardless of grid resolution.
  _injectCircle(center, radiusUv, brushRadius, strength, quadVao) {
    const circumferenceUv = 2 * Math.PI * radiusUv;
    const steps = stepsForUvLength(circumferenceUv, brushRadius);

    for (let step = 0; step < steps; step++) {
      const angle     = (step / steps) * 2 * Math.PI;
      const position  = [
        center[0] + radiusUv * Math.cos(angle),
        center[1] + radiusUv * Math.sin(angle),
      ];
      const direction = [-Math.sin(angle), Math.cos(angle)]; // CCW tangent
      this._fluidField.injectImpulse(position, direction, brushRadius, strength, quadVao);
    }
  }

  // Injects each side of a regular polygon independently.
  // Step count per side is derived from the side's UV-space length so coverage is uniform.
  // The last point of each side is omitted to avoid double-injection at vertices.
  _injectRegularPolygon(center, sides, radiusUv, brushRadius, strength, quadVao) {
    for (let sideIndex = 0; sideIndex < sides; sideIndex++) {
      const angleA  = (sideIndex / sides)       * 2 * Math.PI - Math.PI / 2;
      const angleB  = ((sideIndex + 1) / sides) * 2 * Math.PI - Math.PI / 2;

      const vertexA = [
        center[0] + radiusUv * Math.cos(angleA),
        center[1] + radiusUv * Math.sin(angleA),
      ];
      const vertexB = [
        center[0] + radiusUv * Math.cos(angleB),
        center[1] + radiusUv * Math.sin(angleB),
      ];

      const sideVec      = [vertexB[0] - vertexA[0], vertexB[1] - vertexA[1]];
      const sideLengthUv = Math.hypot(sideVec[0], sideVec[1]);
      const direction    = [sideVec[0] / sideLengthUv, sideVec[1] / sideLengthUv];
      const stepsForSide = stepsForUvLength(sideLengthUv, brushRadius);

      for (let step = 0; step < stepsForSide; step++) {
        const t        = step / stepsForSide; // exclude endpoint → handled by next side
        const position = [
          vertexA[0] + t * sideVec[0],
          vertexA[1] + t * sideVec[1],
        ];
        this._fluidField.injectImpulse(position, direction, brushRadius, strength, quadVao);
      }
    }
  }

  // Injects a filled disk in a single shader pass — no Gaussian point accumulation.
  _injectFilledDisk(center, mode, quadVao) {
    const patternRadius = MOUSE_DEFAULTS.patternScale * 0.5;
    const strength      = MOUSE_DEFAULTS.impulseStrength;
    const modeIndex     = mode === 'spin' ? 0 : mode === 'explode' ? 1 : 2;
    this._fluidField.injectDisk(center, patternRadius, strength, modeIndex, quadVao);
  }

  // At radius ≤ 1 every grid cell gets its own random direction (per-pixel noise).
  // At radius > 1 the field is tiled with overlapping Gaussian impulses (step = 0.75 × radius)
  // each pointing in a freshly randomised direction — coarser, blobby noise.
  _injectNoise(quadVao) {
    const brushRadius = MOUSE_DEFAULTS.impulseRadius;
    const strength    = MOUSE_DEFAULTS.impulseStrength;

    if (brushRadius <= 1) {
      this._fluidField.addNoise(strength, Math.random(), quadVao);
      return;
    }

    const stepUv = (brushRadius * INJECTION_STEP_FRACTION) / GRID_SIZE;
    const cols   = Math.ceil(1 / stepUv) + 1;
    const rows   = Math.ceil(1 / stepUv) + 1;

    for (let row = 0; row < rows; row++) {
      for (let col = 0; col < cols; col++) {
        const u     = col * stepUv;
        const v     = row * stepUv;
        const angle = Math.random() * 2 * Math.PI;
        this._fluidField.injectImpulse(
          [u, v],
          [Math.cos(angle), Math.sin(angle)],
          brushRadius,
          strength,
          quadVao,
        );
      }
    }
  }

  // Four arms of a cross (at 0°/90°/180°/270°), each injected from center outward.
  // spin: CCW tangential velocity along each arm (like wheel spokes).
  // explode: velocity pointing outward along each arm.
  // implode: velocity pointing inward along each arm.
  _injectCross(center, mode, quadVao) {
    const patternRadius = MOUSE_DEFAULTS.patternScale * 0.5;
    const brushRadius   = MOUSE_DEFAULTS.impulseRadius;
    const strength      = MOUSE_DEFAULTS.impulseStrength;

    // Arm directions and their CCW perpendiculars for spin.
    const arms = [
      { armDir: [ 1,  0], spinDir: [ 0,  1] }, // right  → spin up
      { armDir: [ 0,  1], spinDir: [-1,  0] }, // top    → spin left
      { armDir: [-1,  0], spinDir: [ 0, -1] }, // left   → spin down
      { armDir: [ 0, -1], spinDir: [ 1,  0] }, // bottom → spin right
    ];

    const steps = stepsForUvLength(patternRadius, brushRadius);

    for (const { armDir, spinDir } of arms) {
      for (let step = 0; step < steps; step++) {
        const t        = step / (steps - 1);
        const position = [
          center[0] + armDir[0] * t * patternRadius,
          center[1] + armDir[1] * t * patternRadius,
        ];
        const direction =
          mode === 'spin'    ? spinDir :
          mode === 'explode' ? armDir  :
          [-armDir[0], -armDir[1]]; // implode
        this._fluidField.injectImpulse(position, direction, brushRadius, strength, quadVao);
      }
    }
  }

  // Five horizontal stripes with alternating ←→ flow — seeds Kelvin-Helmholtz shear instability.
  _injectStripes(center, quadVao) {
    const STRIPE_COUNT  = 5;
    const patternRadius = MOUSE_DEFAULTS.patternScale * 0.5;
    const HALF_WIDTH    = patternRadius * 1.1;
    const HALF_HEIGHT   = patternRadius;
    const radius        = MOUSE_DEFAULTS.impulseRadius;
    const strength      = MOUSE_DEFAULTS.impulseStrength;

    const pointsPerStripe = stepsForUvLength(2 * HALF_WIDTH, radius);

    for (let row = 0; row < STRIPE_COUNT; row++) {
      const v         = center[1] + HALF_HEIGHT * (row / (STRIPE_COUNT - 1) * 2 - 1);
      const direction = row % 2 === 0 ? [1, 0] : [-1, 0];

      for (let col = 0; col < pointsPerStripe; col++) {
        const t = col / (pointsPerStripe - 1); // [0, 1] inclusive
        const u = center[0] + HALF_WIDTH * (t * 2 - 1);
        this._fluidField.injectImpulse([u, v], direction, radius, strength, quadVao);
      }
    }
  }

  // 5 horizontal + 5 vertical lines, alternating flow direction per line.
  // Crossing orthogonal flows create vortices at every junction.
  _injectSquareGridLines(center, quadVao) {
    const LINE_COUNT    = 5;
    const patternRadius = MOUSE_DEFAULTS.patternScale * 0.5;
    const HALF_SPAN     = patternRadius;
    const radius        = MOUSE_DEFAULTS.impulseRadius;
    const strength      = MOUSE_DEFAULTS.impulseStrength;

    const pointsPerLine = stepsForUvLength(2 * HALF_SPAN, radius);

    for (let k = 0; k < LINE_COUNT; k++) {
      const offset = HALF_SPAN * (k / (LINE_COUNT - 1) * 2 - 1);

      // Horizontal line at y = center_y + offset
      const hDirection = k % 2 === 0 ? [1, 0] : [-1, 0];
      for (let p = 0; p < pointsPerLine; p++) {
        const t = p / (pointsPerLine - 1);
        const x = center[0] + HALF_SPAN * (t * 2 - 1);
        this._fluidField.injectImpulse([x, center[1] + offset], hDirection, radius, strength, quadVao);
      }

      // Vertical line at x = center_x + offset
      const vDirection = k % 2 === 0 ? [0, 1] : [0, -1];
      for (let p = 0; p < pointsPerLine; p++) {
        const t = p / (pointsPerLine - 1);
        const y = center[1] + HALF_SPAN * (t * 2 - 1);
        this._fluidField.injectImpulse([center[0] + offset, y], vDirection, radius, strength, quadVao);
      }
    }
  }

  // Three families of 5 parallel lines at 0°/60°/120°, each flowing along its line direction.
  // Creates hexagonal interference structure.
  _injectTriangularGridLines(center, quadVao) {
    const LINE_COUNT    = 5;
    const patternRadius = MOUSE_DEFAULTS.patternScale * 0.5;
    const HALF_SPAN     = patternRadius * 1.1;
    const HALF_RANGE    = patternRadius;
    const radius        = MOUSE_DEFAULTS.impulseRadius;
    const strength      = MOUSE_DEFAULTS.impulseStrength;

    const pointsPerLine = stepsForUvLength(2 * HALF_SPAN, radius);
    const familyAngles  = [0, Math.PI / 3, 2 * Math.PI / 3];

    for (const angle of familyAngles) {
      const direction = [Math.cos(angle), Math.sin(angle)];
      const normal    = [-Math.sin(angle), Math.cos(angle)];

      for (let k = 0; k < LINE_COUNT; k++) {
        const offset = HALF_RANGE * (k / (LINE_COUNT - 1) * 2 - 1);

        for (let p = 0; p < pointsPerLine; p++) {
          const t = p / (pointsPerLine - 1);
          const s = HALF_SPAN * (t * 2 - 1);
          const u = center[0] + s * direction[0] + offset * normal[0];
          const v = center[1] + s * direction[1] + offset * normal[1];
          this._fluidField.injectImpulse([u, v], direction, radius, strength, quadVao);
        }
      }
    }
  }

  // Hexagonally packed points (center + ring of 6 + ring of 12), each pointing radially outward.
  // These are discrete source points so step count does not apply.
  _injectScatteredPoints(center, quadVao) {
    const INNER_COUNT   = 6;
    const patternRadius = MOUSE_DEFAULTS.patternScale * 0.5;
    const INNER_RADIUS  = patternRadius * 0.5;
    const OUTER_COUNT   = 12;
    const OUTER_RADIUS  = patternRadius;
    const radius        = MOUSE_DEFAULTS.impulseRadius;
    const strength      = MOUSE_DEFAULTS.impulseStrength;

    this._fluidField.injectImpulse(center, [1, 0], radius * 1.5, strength, quadVao);

    for (let i = 0; i < INNER_COUNT; i++) {
      const angle    = (i / INNER_COUNT) * 2 * Math.PI;
      const position = [
        center[0] + INNER_RADIUS * Math.cos(angle),
        center[1] + INNER_RADIUS * Math.sin(angle),
      ];
      this._fluidField.injectImpulse(position, [Math.cos(angle), Math.sin(angle)], radius, strength, quadVao);
    }

    for (let i = 0; i < OUTER_COUNT; i++) {
      const angle    = (i / OUTER_COUNT) * 2 * Math.PI;
      const position = [
        center[0] + OUTER_RADIUS * Math.cos(angle),
        center[1] + OUTER_RADIUS * Math.sin(angle),
      ];
      this._fluidField.injectImpulse(position, [Math.cos(angle), Math.sin(angle)], radius, strength, quadVao);
    }
  }
}

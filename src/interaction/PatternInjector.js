'use strict';

import { MOUSE_DEFAULTS } from '../config/SimulationConfig.js';

const PATTERN_RADIUS = 0.28;  // half-extent of shape patterns in UV [0,1] space

const PATTERNS = [
  { value: 'circle',    label: 'Circle'               },
  { value: 'triangle',  label: 'Triangle'             },
  { value: 'square',    label: 'Square'               },
  { value: 'pentagon',  label: 'Pentagon'             },
  { value: 'hexagon',   label: 'Hexagon'              },
  { value: 'heptagon',  label: 'Heptagon'             },
  { value: 'octagon',   label: 'Octagon'              },
  { value: 'nonagon',   label: 'Nonagon'              },
  { value: 'decagon',   label: 'Decagon'              },
  { value: 'stripes',   label: 'Parallel stripes'     },
  { value: 'gridlines', label: 'Square grid lines'    },
  { value: 'trilines',  label: 'Triangular grid lines'},
  { value: 'points',    label: 'Scattered points'     },
  { value: 'noise',     label: 'Random noise'         },
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
      this._fluidField.addNoise(MOUSE_DEFAULTS.impulseStrength, Math.random(), quadVao);
      return;
    }

    const sides = POLYGON_SIDES[action];
    if (sides !== undefined) {
      this._injectPolygon(center, sides, quadVao);
      return;
    }
    if (action === 'stripes')   this._injectStripes(center, quadVao);
    if (action === 'gridlines') this._injectSquareGridLines(center, quadVao);
    if (action === 'trilines')  this._injectTriangularGridLines(center, quadVao);
    if (action === 'points')    this._injectScatteredPoints(center, quadVao);
  }

  // Injects 32 points along a regular polygon's perimeter with CCW tangential velocity.
  // sides=0 produces a circle.
  _injectPolygon(center, sides, quadVao) {
    const STEPS    = 32;
    const isCircle = sides < 3;
    const radius   = MOUSE_DEFAULTS.impulseRadius;
    const strength = MOUSE_DEFAULTS.impulseStrength;

    for (let step = 0; step < STEPS; step++) {
      const t = step / STEPS;
      let position, direction;

      if (isCircle) {
        const angle = t * 2 * Math.PI;
        position  = [
          center[0] + PATTERN_RADIUS * Math.cos(angle),
          center[1] + PATTERN_RADIUS * Math.sin(angle),
        ];
        direction = [-Math.sin(angle), Math.cos(angle)];
      } else {
        // Distribute t uniformly across N sides; regular polygon so arc-length is uniform too.
        const sideIndex = Math.floor(t * sides);
        const sideT     = t * sides - sideIndex;
        const angleA    = (sideIndex / sides)       * 2 * Math.PI - Math.PI / 2;
        const angleB    = ((sideIndex + 1) / sides) * 2 * Math.PI - Math.PI / 2;
        const vA = [
          center[0] + PATTERN_RADIUS * Math.cos(angleA),
          center[1] + PATTERN_RADIUS * Math.sin(angleA),
        ];
        const vB = [
          center[0] + PATTERN_RADIUS * Math.cos(angleB),
          center[1] + PATTERN_RADIUS * Math.sin(angleB),
        ];
        position = [
          vA[0] + sideT * (vB[0] - vA[0]),
          vA[1] + sideT * (vB[1] - vA[1]),
        ];
        const sideLen = Math.hypot(vB[0] - vA[0], vB[1] - vA[1]);
        direction = [(vB[0] - vA[0]) / sideLen, (vB[1] - vA[1]) / sideLen];
      }

      this._fluidField.injectImpulse(position, direction, radius, strength, quadVao);
    }
  }

  // Five horizontal stripes with alternating left/right flow — seeds Kelvin-Helmholtz shear.
  _injectStripes(center, quadVao) {
    const STRIPE_COUNT      = 5;
    const POINTS_PER_STRIPE = 8;
    const HALF_WIDTH        = PATTERN_RADIUS * 1.1;
    const HALF_HEIGHT       = PATTERN_RADIUS;
    const radius            = MOUSE_DEFAULTS.impulseRadius;
    const strength          = MOUSE_DEFAULTS.impulseStrength;

    for (let row = 0; row < STRIPE_COUNT; row++) {
      const v         = center[1] + HALF_HEIGHT * (row / (STRIPE_COUNT - 1) * 2 - 1);
      const direction = row % 2 === 0 ? [1, 0] : [-1, 0];
      for (let col = 0; col < POINTS_PER_STRIPE; col++) {
        const u = center[0] + HALF_WIDTH * (col / (POINTS_PER_STRIPE - 1) * 2 - 1);
        this._fluidField.injectImpulse([u, v], direction, radius, strength, quadVao);
      }
    }
  }

  // 5 horizontal + 5 vertical lines of 7 points each, alternating flow direction per line.
  // Crossing orthogonal flows create vortices at every junction.
  _injectSquareGridLines(center, quadVao) {
    const LINE_COUNT      = 5;
    const POINTS_PER_LINE = 7;
    const HALF_SPAN       = PATTERN_RADIUS;
    const radius          = MOUSE_DEFAULTS.impulseRadius;
    const strength        = MOUSE_DEFAULTS.impulseStrength;

    for (let k = 0; k < LINE_COUNT; k++) {
      const offset = HALF_SPAN * (k / (LINE_COUNT - 1) * 2 - 1);

      // Horizontal line at y = center_y + offset
      const hDir = k % 2 === 0 ? [1, 0] : [-1, 0];
      for (let p = 0; p < POINTS_PER_LINE; p++) {
        const t = HALF_SPAN * (p / (POINTS_PER_LINE - 1) * 2 - 1);
        this._fluidField.injectImpulse([center[0] + t, center[1] + offset], hDir, radius, strength, quadVao);
      }

      // Vertical line at x = center_x + offset
      const vDir = k % 2 === 0 ? [0, 1] : [0, -1];
      for (let p = 0; p < POINTS_PER_LINE; p++) {
        const t = HALF_SPAN * (p / (POINTS_PER_LINE - 1) * 2 - 1);
        this._fluidField.injectImpulse([center[0] + offset, center[1] + t], vDir, radius, strength, quadVao);
      }
    }
  }

  // Three families of 5 parallel lines at 0°, 60°, 120°, each flowing along its line direction.
  // Creates a triangular lattice with hexagonal interference structure.
  _injectTriangularGridLines(center, quadVao) {
    const LINE_COUNT      = 5;
    const POINTS_PER_LINE = 7;
    const HALF_SPAN       = PATTERN_RADIUS * 1.1;
    const HALF_RANGE      = PATTERN_RADIUS;
    const radius          = MOUSE_DEFAULTS.impulseRadius;
    const strength        = MOUSE_DEFAULTS.impulseStrength;

    const familyAngles = [0, Math.PI / 3, 2 * Math.PI / 3];

    for (const angle of familyAngles) {
      const dir    = [Math.cos(angle), Math.sin(angle)];
      const normal = [-Math.sin(angle), Math.cos(angle)];

      for (let k = 0; k < LINE_COUNT; k++) {
        const offset = HALF_RANGE * (k / (LINE_COUNT - 1) * 2 - 1);

        for (let p = 0; p < POINTS_PER_LINE; p++) {
          const t = HALF_SPAN * (p / (POINTS_PER_LINE - 1) * 2 - 1);
          const u = center[0] + t * dir[0] + offset * normal[0];
          const v = center[1] + t * dir[1] + offset * normal[1];
          this._fluidField.injectImpulse([u, v], dir, radius, strength, quadVao);
        }
      }
    }
  }

  // Hexagonally packed points (center + ring of 6 + ring of 12), each pointing radially outward.
  _injectScatteredPoints(center, quadVao) {
    const INNER_COUNT  = 6;
    const INNER_RADIUS = PATTERN_RADIUS * 0.5;
    const OUTER_COUNT  = 12;
    const OUTER_RADIUS = PATTERN_RADIUS;
    const radius       = MOUSE_DEFAULTS.impulseRadius;
    const strength     = MOUSE_DEFAULTS.impulseStrength;

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

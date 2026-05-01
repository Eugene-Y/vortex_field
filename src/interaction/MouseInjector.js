'use strict';

import { MOUSE_DEFAULTS, MOUSE_SPEED_REFERENCE, GRID_SIZE } from '../config/SimulationConfig.js';

const INJECTION_STEP_FRACTION = 0.75; // same coverage rule as PatternInjector

/**
 * Translates mouse events on the velocity half of the canvas into impulse
 * injection calls on a FluidField.
 *
 * The canvas holds two fields side by side. Only the left half (width = fieldSize)
 * maps to the velocity field. Clicks on the right half are ignored.
 *
 * Position is normalized to [0,1] UV space. Direction is derived from mouse delta.
 * Between frames, the full stroke segment from previous to current position is
 * interpolated so fast mouse movement leaves no gaps.
 */
export class MouseInjector {
  constructor(canvas, fluidField, fieldSize) {
    this._canvas    = canvas;
    this._fluidField = fluidField;
    this._fieldSize  = fieldSize;
    this._pressedInField    = false; // true only when mousedown originated on Field A
    this._previousPosition  = null;
    this._previousEventTime = null;
    this._pendingStroke     = null; // { from, to, direction, speedFactor }

    this._onMouseDown = this._onMouseDown.bind(this);
    this._onMouseMove = this._onMouseMove.bind(this);
    this._onMouseUp   = this._onMouseUp.bind(this);

    canvas.addEventListener('mousedown', this._onMouseDown);
    canvas.addEventListener('mousemove', this._onMouseMove);
    window.addEventListener('mouseup',   this._onMouseUp);
  }

  applyPendingInjection() {
    if (!this._pendingStroke) return;
    const { from, to, direction, speedFactor } = this._pendingStroke;
    this._pendingStroke = null;
    this._injectStroke(from, to, direction, speedFactor);
  }

  dispose() {
    this._canvas.removeEventListener('mousedown', this._onMouseDown);
    this._canvas.removeEventListener('mousemove', this._onMouseMove);
    window.removeEventListener('mouseup', this._onMouseUp);
  }

  _injectStroke(from, to, direction, speedFactor) {
    const dx = to[0] - from[0];
    const dy = to[1] - from[1];
    const lengthUv = Math.hypot(dx, dy);

    const sensitivity = MOUSE_DEFAULTS.speedSensitivity;
    const strength = MOUSE_DEFAULTS.impulseStrength * (1 - sensitivity + sensitivity * speedFactor);

    const stepUv = (MOUSE_DEFAULTS.impulseRadius * INJECTION_STEP_FRACTION) / GRID_SIZE;
    const steps  = Math.max(1, Math.ceil(lengthUv / stepUv));

    for (let i = 0; i < steps; i++) {
      const t = steps === 1 ? 0 : i / (steps - 1);
      const position = [from[0] + t * dx, from[1] + t * dy];
      this._fluidField.injectImpulse(
        position, direction,
        MOUSE_DEFAULTS.impulseRadius,
        strength,
      );
    }
  }

  _onMouseDown(event) {
    const position = this._normalizeToFieldUv(event);
    if (!position) return;
    this._pressedInField    = true;
    this._previousPosition  = position;
    this._previousEventTime = event.timeStamp;
  }

  _onMouseMove(event) {
    if (!this._pressedInField) return;

    const currentPosition = this._normalizeToFieldUv(event);

    if (!currentPosition) {
      // Mouse left Field A — forget previous position so re-entry doesn't
      // create a phantom stroke spanning the gap.
      this._previousPosition  = null;
      this._previousEventTime = null;
      return;
    }

    if (!this._previousPosition) {
      this._previousPosition  = currentPosition;
      this._previousEventTime = event.timeStamp;
      return;
    }

    const dx = currentPosition[0] - this._previousPosition[0];
    const dy = currentPosition[1] - this._previousPosition[1];
    const speed = Math.hypot(dx, dy);

    if (speed > 0) {
      const pixelDelta      = speed * this._fieldSize;
      const deltaSeconds    = Math.max(0.001, (event.timeStamp - this._previousEventTime) / 1000);
      const pixelsPerSecond = pixelDelta / deltaSeconds;
      const speedFactor     = pixelsPerSecond / MOUSE_SPEED_REFERENCE;

      this._pendingStroke = {
        from:        this._previousPosition,
        to:          currentPosition,
        direction:   [dx / speed, dy / speed],
        speedFactor,
      };
    }

    this._previousPosition  = currentPosition;
    this._previousEventTime = event.timeStamp;
  }

  _onMouseUp() {
    this._pressedInField    = false;
    this._previousPosition  = null;
    this._previousEventTime = null;
  }

  _normalizeToFieldUv(event) {
    const rect   = this._canvas.getBoundingClientRect();
    const scaleX = this._canvas.width  / rect.width;
    const scaleY = this._canvas.height / rect.height;

    const pixelX = (event.clientX - rect.left) * scaleX;
    const pixelY = (event.clientY - rect.top)  * scaleY;

    if (pixelX < 0 || pixelX > this._fieldSize) return null;
    if (pixelY < 0 || pixelY > this._fieldSize) return null;

    return [
      pixelX / this._fieldSize,
      1.0 - pixelY / this._fieldSize,
    ];
  }
}

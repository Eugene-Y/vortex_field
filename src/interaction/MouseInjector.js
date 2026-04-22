'use strict';

import { MOUSE_DEFAULTS } from '../config/SimulationConfig.js';

/**
 * Translates mouse events on the velocity half of the canvas into impulse
 * injection calls on a FluidField.
 *
 * The canvas holds two fields side by side. Only the left half (width = fieldSize)
 * maps to the velocity field. Clicks on the right half are ignored.
 *
 * Position is normalized to [0,1] UV space. Direction is derived from mouse delta.
 */
export class MouseInjector {
  constructor(canvas, fluidField, fieldSize) {
    this._canvas = canvas;
    this._fluidField = fluidField;
    this._fieldSize = fieldSize;
    this._isPressed = false;
    this._previousPosition = null;
    this._pendingInjection = null;

    this._onMouseDown = this._onMouseDown.bind(this);
    this._onMouseMove = this._onMouseMove.bind(this);
    this._onMouseUp   = this._onMouseUp.bind(this);

    canvas.addEventListener('mousedown', this._onMouseDown);
    canvas.addEventListener('mousemove', this._onMouseMove);
    window.addEventListener('mouseup',   this._onMouseUp);
  }

  /**
   * Called each frame by the render loop. Applies any pending impulse injection.
   */
  applyPendingInjection(quadVao) {
    if (!this._pendingInjection) return;

    const { position, direction, radius, strength } = this._pendingInjection;
    this._pendingInjection = null;
    this._fluidField.injectImpulse(position, direction, radius, strength, quadVao);
  }

  dispose() {
    this._canvas.removeEventListener('mousedown', this._onMouseDown);
    this._canvas.removeEventListener('mousemove', this._onMouseMove);
    window.removeEventListener('mouseup', this._onMouseUp);
  }

  _onMouseDown(event) {
    const position = this._normalizeToFieldUv(event);
    if (!position) return;
    this._isPressed = true;
    this._previousPosition = position;
  }

  _onMouseMove(event) {
    if (!this._isPressed) return;

    const currentPosition = this._normalizeToFieldUv(event);
    if (!currentPosition) return;

    const direction = [
      currentPosition[0] - this._previousPosition[0],
      currentPosition[1] - this._previousPosition[1],
    ];
    const speed = Math.hypot(direction[0], direction[1]);

    if (speed > 0) {
      this._pendingInjection = {
        position: currentPosition,
        direction: [direction[0] / speed, direction[1] / speed],
        radius: MOUSE_DEFAULTS.impulseRadius,
        strength: MOUSE_DEFAULTS.impulseStrength,
      };
    }

    this._previousPosition = currentPosition;
  }

  _onMouseUp() {
    this._isPressed = false;
    this._previousPosition = null;
  }

  /**
   * Returns normalized [0,1] UV coordinates within the velocity field half,
   * or null if the cursor is outside that half.
   */
  _normalizeToFieldUv(event) {
    const rect = this._canvas.getBoundingClientRect();
    const scaleX = this._canvas.width  / rect.width;
    const scaleY = this._canvas.height / rect.height;

    const pixelX = (event.clientX - rect.left) * scaleX;
    const pixelY = (event.clientY - rect.top)  * scaleY;

    if (pixelX > this._fieldSize) return null; // right half = rotation field, not interactive

    return [
      pixelX / this._fieldSize,
      1.0 - pixelY / this._fieldSize, // flip Y: WebGL UV origin is bottom-left
    ];
  }
}

'use strict';

import { ROTATION_FIELD, MOUSE_DEFAULTS, GRID_SIZE, buildShareUrl } from '../config/SimulationConfig.js';

const DRAG_MOVED_THRESHOLD_PX = 5;

/**
 * Manages the focus-mask circle on Field B.
 *
 * Mousedown on Field B activates the mask at that grid position.
 * Dragging moves it in real time. A quick click (no drag) on an
 * already-active mask clears it. Escape always clears.
 *
 * The mask radius tracks MOUSE_DEFAULTS.impulseRadius so the brush
 * radius slider controls the focus circle size.
 *
 * An overlay canvas (Canvas 2D) is created above Field A to display
 * the active circle. Call updateOverlay() each frame.
 */
export class FocusMaskInteractor {
  constructor(fieldBCanvas, fieldACanvas) {
    this._fieldBCanvas  = fieldBCanvas;
    this._fieldACanvas  = fieldACanvas;
    this._dragging      = false;
    this._dragMoved     = false;
    this._startedOnMask = false;
    this._dragStartPx   = null;

    this._overlay = this._createOverlayCanvas(fieldACanvas);
    this._ctx     = this._overlay.getContext('2d');

    fieldBCanvas.addEventListener('mousedown',   this._onMouseDown.bind(this));
    window.addEventListener('mousemove',         this._onMouseMove.bind(this));
    window.addEventListener('mouseup',           this._onMouseUp.bind(this));
    window.addEventListener('keydown',           this._onKeyDown.bind(this));
  }

  // Must be called each frame so the overlay reflects slider changes to brush radius.
  updateOverlay() {
    this._drawOverlay();
  }

  _createOverlayCanvas(fieldACanvas) {
    const overlay = document.createElement('canvas');
    overlay.width  = fieldACanvas.width;
    overlay.height = fieldACanvas.height;
    overlay.style.cssText = 'position:absolute;top:0;left:0;pointer-events:none;';
    const wrapper = fieldACanvas.parentElement;
    wrapper.style.position = 'relative';
    wrapper.insertBefore(overlay, fieldACanvas.nextSibling);
    return overlay;
  }

  _canvasEventToGridCoords(event) {
    const rect = this._fieldBCanvas.getBoundingClientRect();
    const col  = ((event.clientX - rect.left) / rect.width)  * GRID_SIZE;
    // WebGL texture row 0 is at the bottom; canvas Y=0 is at the top.
    const row  = (1 - (event.clientY - rect.top) / rect.height) * GRID_SIZE;
    return [col, row];
  }

  _distanceToCurrentMaskCenter(candidate) {
    const center = ROTATION_FIELD.maskCenter;
    if (!center) return Infinity;
    const dc = candidate[0] - center[0];
    const dr = candidate[1] - center[1];
    return Math.sqrt(dc * dc + dr * dr);
  }

  _onMouseDown(event) {
    if (event.button !== 0) return;
    event.preventDefault();
    const coords = this._canvasEventToGridCoords(event);
    this._dragging      = true;
    this._dragMoved     = false;
    this._dragStartPx   = [event.clientX, event.clientY];
    this._startedOnMask = this._distanceToCurrentMaskCenter(coords) <= MOUSE_DEFAULTS.impulseRadius;
    this._setMask(coords);
  }

  _onMouseMove(event) {
    if (!this._dragging) return;
    const dx = event.clientX - this._dragStartPx[0];
    const dy = event.clientY - this._dragStartPx[1];
    if (!this._dragMoved && Math.sqrt(dx * dx + dy * dy) > DRAG_MOVED_THRESHOLD_PX) {
      this._dragMoved = true;
    }
    if (this._dragMoved) {
      this._setMask(this._canvasEventToGridCoords(event));
    }
  }

  _onMouseUp(event) {
    if (!this._dragging) return;
    this._dragging = false;
    if (this._startedOnMask && !this._dragMoved) {
      this._clearMask();
    } else {
      this._commitMaskToUrl();
    }
  }

  _onKeyDown(event) {
    if (event.key === 'Escape') this._clearMask();
  }

  _setMask(coords) {
    ROTATION_FIELD.maskCenter = coords;
    ROTATION_FIELD.maskRadius = MOUSE_DEFAULTS.impulseRadius;
  }

  _commitMaskToUrl() {
    history.replaceState(null, '', buildShareUrl());
  }

  _clearMask() {
    ROTATION_FIELD.maskCenter = null;
    ROTATION_FIELD.maskRadius = null;
    this._commitMaskToUrl();
    this._drawOverlay();
  }

  _drawOverlay() {
    const ctx    = this._ctx;
    const size   = this._overlay.width;
    ctx.clearRect(0, 0, size, size);

    const center = ROTATION_FIELD.maskCenter;
    if (!center) return;

    const radius = ROTATION_FIELD.maskRadius ?? MOUSE_DEFAULTS.impulseRadius;
    const px = center[0] / GRID_SIZE * size;
    const py = (1 - center[1] / GRID_SIZE) * size;  // flip: texture row → screen Y
    const pr = radius    / GRID_SIZE * size;

    ctx.beginPath();
    ctx.arc(px, py, Math.max(1, pr), 0, Math.PI * 2);
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.8)';
    ctx.lineWidth   = 2.5;
    ctx.setLineDash([5, 4]);
    ctx.stroke();
  }
}

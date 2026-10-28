'use strict';

import { GRID_SIZE } from '../config/SimulationConfig.js';

// Physical key, so the shortcut works whatever keyboard layout is active.
const SAVE_KEY_CODE = 'ArrowDown';
const SNAPSHOT_TYPE = 'image/png';

const twoDigits = value => String(value).padStart(2, '0');

// Local time as YYYYMMDD_HHMMSS.
function formatTimestamp(date) {
  const day  = `${date.getFullYear()}${twoDigits(date.getMonth() + 1)}${twoDigits(date.getDate())}`;
  const time = `${twoDigits(date.getHours())}${twoDigits(date.getMinutes())}${twoDigits(date.getSeconds())}`;
  return `${day}_${time}`;
}

// Some browsers start the download after click() returns; revoking at once can cancel it.
const OBJECT_URL_LIFETIME_MS = 10000;

function downloadBlob(blob, fileName) {
  const url  = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href     = url;
  link.download = fileName;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), OBJECT_URL_LIFETIME_MS);
}

/**
 * Saves both fields as PNG files when the down arrow is pressed:
 * vortex_a_YYYYMMDD_HHMMSS.png and vortex_b_YYYYMMDD_HHMMSS.png, one pixel per grid cell.
 *
 * A WebGPU canvas only holds its image until the frame is presented, so the key just
 * requests a snapshot; saveIfRequested() must run right after both fields are rendered,
 * in the same task. While the simulation is paused, requestRender() is called so that a
 * fresh frame exists to read.
 */
export class FieldSnapshotSaver {
  constructor(canvasVelocity, canvasRotation, requestRender) {
    this._canvasVelocity = canvasVelocity;
    this._canvasRotation = canvasRotation;
    this._requestRender  = requestRender;
    this._requested      = false;

    this._onKeyDown = this._onKeyDown.bind(this);
    window.addEventListener('keydown', this._onKeyDown);
  }

  saveIfRequested() {
    if (!this._requested) return;
    this._requested = false;
    const timestamp = formatTimestamp(new Date());
    this._saveCanvas(this._canvasVelocity, `vortex_a_${timestamp}.png`);
    this._saveCanvas(this._canvasRotation, `vortex_b_${timestamp}.png`);
  }

  dispose() {
    window.removeEventListener('keydown', this._onKeyDown);
  }

  _onKeyDown(event) {
    if (event.code !== SAVE_KEY_CODE || event.repeat) return;
    if (event.metaKey || event.ctrlKey || event.altKey || event.shiftKey) return;
    if (this._isTypingIntoField(event.target)) return;
    event.preventDefault();
    this._requested = true;
    this._requestRender();
  }

  _isTypingIntoField(target) {
    return target.matches?.('input, select, textarea, [contenteditable]') ?? false;
  }

  // The canvas shows each cell as a block of identical pixels, so nearest-neighbour
  // downscaling to the grid size picks exactly one pixel per cell. drawImage copies the
  // WebGPU canvas synchronously; only the PNG encoding is deferred.
  _saveCanvas(canvas, fileName) {
    const gridImage = new OffscreenCanvas(GRID_SIZE, GRID_SIZE);
    const context   = gridImage.getContext('2d');
    context.imageSmoothingEnabled = false;
    context.drawImage(canvas, 0, 0, GRID_SIZE, GRID_SIZE);
    gridImage.convertToBlob({ type: SNAPSHOT_TYPE }).then(blob => downloadBlob(blob, fileName));
  }
}

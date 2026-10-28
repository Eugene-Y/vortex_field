'use strict';

import { MOUSE_DEFAULTS, MOUSE_SPEED_REFERENCE, GRID_SIZE, BRUSH } from '../config/SimulationConfig.js';

const INJECTION_STEP_FRACTION = 0.75; // same coverage rule as PatternInjector

// Centripetal Catmull–Rom (α = 0.5) passes through every sample without cusps or
// overshoot loops, even when samples are unevenly spaced.
const CENTRIPETAL_ALPHA = 0.5;

// Samples closer than this to the previous one (in grid cells) carry no direction.
const MIN_SAMPLE_SPACING_CELLS = 0.25;

// Points per segment used to estimate its arc length for stamp spacing.
const ARC_LENGTH_SAMPLES = 8;

// Parameter step for the finite-difference tangent of the curve.
const TANGENT_STEP = 1e-3;

const MIN_DELTA_SECONDS = 0.001;

/**
 * Translates pointer strokes on Field A into impulse injections on a FluidField.
 *
 * Every pointer sample is used, including the ones the browser coalesces between frames,
 * and the stroke follows a smooth curve through them. A slow frame therefore no longer
 * turns the stroke into a polyline whose impulse direction jumps at each corner.
 *
 * A segment between two samples is queued once the following sample arrives (its end
 * tangent needs it); the last segment is queued when the stroke ends. All segments
 * queued since the previous frame are injected by applyPendingInjection().
 */
export class MouseInjector {
  constructor(canvas, fluidField, fieldSize) {
    this._canvas     = canvas;
    this._fluidField = fluidField;
    this._fieldSize  = fieldSize;
    this._pressedInField  = false; // true only when the press originated on Field A
    this._strokeSamples   = [];    // last few { position, time } of the current stroke
    this._pendingSegments = [];

    this._onPointerDown = this._onPointerDown.bind(this);
    this._onPointerMove = this._onPointerMove.bind(this);
    this._onPointerUp   = this._onPointerUp.bind(this);

    canvas.addEventListener('pointerdown', this._onPointerDown);
    canvas.addEventListener('pointermove', this._onPointerMove);
    window.addEventListener('pointerup',   this._onPointerUp);
  }

  applyPendingInjection() {
    const segments = this._pendingSegments;
    this._pendingSegments = [];
    for (const segment of segments) this._injectSegment(segment);
  }

  dispose() {
    this._canvas.removeEventListener('pointerdown', this._onPointerDown);
    this._canvas.removeEventListener('pointermove', this._onPointerMove);
    window.removeEventListener('pointerup', this._onPointerUp);
  }

  _onPointerDown(event) {
    const position = this._normalizeToFieldUv(event);
    if (!position) return;
    this._pressedInField = true;
    this._strokeSamples  = [{ position, time: event.timeStamp }];
  }

  _onPointerMove(event) {
    if (!this._pressedInField) return;
    const samples = event.getCoalescedEvents?.() ?? [];
    for (const sample of samples.length > 0 ? samples : [event]) this._addPointerSample(sample);
  }

  _onPointerUp() {
    if (!this._pressedInField) return;
    this._finishStroke();
    this._pressedInField = false;
  }

  // Leaving Field A ends the stroke, so re-entry starts a new one instead of a
  // phantom segment spanning the gap.
  _addPointerSample(event) {
    const position = this._normalizeToFieldUv(event);
    if (!position) {
      this._finishStroke();
      return;
    }
    const samples = this._strokeSamples;
    const last    = samples[samples.length - 1];
    if (!last) {
      samples.push({ position, time: event.timeStamp });
      return;
    }
    if (Math.hypot(position[0] - last.position[0], position[1] - last.position[1]) * GRID_SIZE < MIN_SAMPLE_SPACING_CELLS) return;

    samples.push({ position, time: event.timeStamp });
    if (samples.length >= 3) this._queueSegmentBefore(samples.length - 2, false);
    if (samples.length > 4) samples.shift();
  }

  _finishStroke() {
    const samples = this._strokeSamples;
    if (samples.length >= 2) this._queueSegmentBefore(samples.length - 1, true);
    this._strokeSamples = [];
  }

  // Queues the segment ending at samples[endIndex]. Missing neighbours at the stroke ends
  // are mirrored, which makes the curve start and stop along the chord.
  _queueSegmentBefore(endIndex, isLastSegment) {
    const samples = this._strokeSamples;
    const start   = samples[endIndex - 1];
    const end     = samples[endIndex];
    const before  = samples[endIndex - 2]?.position ?? mirrorPoint(start.position, end.position);
    const after   = samples[endIndex + 1]?.position ?? mirrorPoint(end.position, start.position);

    const pixelDistance   = Math.hypot(end.position[0] - start.position[0], end.position[1] - start.position[1]) * this._fieldSize;
    const deltaSeconds    = Math.max(MIN_DELTA_SECONDS, (end.time - start.time) / 1000);
    const speedFactor     = pixelDistance / deltaSeconds / MOUSE_SPEED_REFERENCE;

    this._pendingSegments.push({
      curve: createCentripetalSegment(before, start.position, end.position, after),
      speedFactor,
      isLastSegment,
    });
  }

  // Stamps are spaced evenly along the segment; the end point is left to the next
  // segment so joints are not stamped twice.
  _injectSegment({ curve, speedFactor, isLastSegment }) {
    const sensitivity = MOUSE_DEFAULTS.speedSensitivity;
    const strength    = MOUSE_DEFAULTS.impulseStrength * (1 - sensitivity + sensitivity * speedFactor);
    const stepUv      = (MOUSE_DEFAULTS.impulseRadius * INJECTION_STEP_FRACTION) / GRID_SIZE;
    const steps       = Math.max(1, Math.ceil(estimateArcLength(curve) / stepUv));
    const lastStep    = isLastSegment ? steps : steps - 1;

    for (let step = 0; step <= lastStep; step++) {
      const t        = step / steps;
      const position = curve.pointAt(t);
      if (BRUSH.mode === 'mass') {
        this._fluidField.injectDensityImpulse(position, MOUSE_DEFAULTS.impulseRadius, strength);
      } else {
        this._fluidField.injectImpulse(position, curve.directionAt(t), MOUSE_DEFAULTS.impulseRadius, strength);
      }
    }
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

function mirrorPoint(point, about) {
  return [2 * point[0] - about[0], 2 * point[1] - about[1]];
}

function lerpPoint(a, b, weightB) {
  return [a[0] + (b[0] - a[0]) * weightB, a[1] + (b[1] - a[1]) * weightB];
}

// Knot spacing grows as distance^α; never zero, so coincident points cannot divide by zero.
function knotInterval(a, b) {
  return Math.max(1e-9, Math.pow(Math.hypot(b[0] - a[0], b[1] - a[1]), CENTRIPETAL_ALPHA));
}

// Curve from p1 to p2 with neighbours p0 and p3, evaluated by the Barry–Goldman pyramid.
// pointAt(t) and directionAt(t) take t ∈ [0, 1] across the p1 → p2 span.
function createCentripetalSegment(p0, p1, p2, p3) {
  const k0 = 0;
  const k1 = k0 + knotInterval(p0, p1);
  const k2 = k1 + knotInterval(p1, p2);
  const k3 = k2 + knotInterval(p2, p3);

  const pointAt = t => {
    const k  = k1 + (k2 - k1) * Math.min(1, Math.max(0, t));
    const a1 = lerpPoint(p0, p1, (k - k0) / (k1 - k0));
    const a2 = lerpPoint(p1, p2, (k - k1) / (k2 - k1));
    const a3 = lerpPoint(p2, p3, (k - k2) / (k3 - k2));
    const b1 = lerpPoint(a1, a2, (k - k0) / (k2 - k0));
    const b2 = lerpPoint(a2, a3, (k - k1) / (k3 - k1));
    return lerpPoint(b1, b2, (k - k1) / (k2 - k1));
  };

  const chord = [p2[0] - p1[0], p2[1] - p1[1]];
  const directionAt = t => {
    const ahead  = pointAt(t + TANGENT_STEP);
    const behind = pointAt(t - TANGENT_STEP);
    const tangent = [ahead[0] - behind[0], ahead[1] - behind[1]];
    const length  = Math.hypot(tangent[0], tangent[1]);
    if (length > 0) return [tangent[0] / length, tangent[1] / length];
    const chordLength = Math.hypot(chord[0], chord[1]);
    return [chord[0] / chordLength, chord[1] / chordLength];
  };

  return { pointAt, directionAt };
}

function estimateArcLength(curve) {
  let length   = 0;
  let previous = curve.pointAt(0);
  for (let sample = 1; sample <= ARC_LENGTH_SAMPLES; sample++) {
    const point = curve.pointAt(sample / ARC_LENGTH_SAMPLES);
    length  += Math.hypot(point[0] - previous[0], point[1] - previous[1]);
    previous = point;
  }
  return length;
}

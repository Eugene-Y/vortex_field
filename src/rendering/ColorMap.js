'use strict';

import { COLORS } from '../config/SimulationConfig.js';

/**
 * Maps a signed rotation scalar to an RGB triple.
 * Positive → orange (CCW), negative → blue (CW), zero → black.
 * Brightness encodes magnitude.
 */
export function mapRotationToColor(rotation) {
  const magnitude = Math.min(Math.abs(rotation), 1.0);
  if (rotation > 0) {
    return scaleColor(COLORS.rotationPositive, magnitude);
  } else if (rotation < 0) {
    return scaleColor(COLORS.rotationNegative, magnitude);
  }
  return COLORS.rotationZero;
}

function scaleColor(color, scale) {
  return [color[0] * scale, color[1] * scale, color[2] * scale];
}

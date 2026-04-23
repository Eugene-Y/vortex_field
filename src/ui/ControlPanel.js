'use strict';

import { RENDER_DEFAULTS } from '../config/SimulationConfig.js';

const LOG_RANGE = 3; // slider covers ±e^3 ≈ 20x around the default value

/**
 * Builds brightness sliders for Field A and Field B.
 * Each slider is logarithmically mapped: center = current default value,
 * left = 20x darker, right = 20x brighter.
 */
export class ControlPanel {
  constructor(velocityContainer, rotationContainer, fieldSize, gapSize) {
    velocityContainer.style.width = `${fieldSize}px`;
    rotationContainer.style.width = `${fieldSize}px`;
    velocityContainer.parentElement.style.gap = `${gapSize}px`;

    this._addBrightnessSlider(velocityContainer, 'Brightness', RENDER_DEFAULTS, 'velocityToneMidpoint');
    this._addBrightnessSlider(rotationContainer, 'Brightness', RENDER_DEFAULTS, 'rotationToneMidpoint');
  }

  _addBrightnessSlider(container, label, config, key) {
    const defaultMidpoint = config[key];

    const wrapper = document.createElement('div');
    wrapper.className = 'control-row';

    const labelEl = document.createElement('span');
    labelEl.className = 'control-label';
    labelEl.textContent = label;

    const slider = document.createElement('input');
    slider.type = 'range';
    slider.min = 0;
    slider.max = 100;
    slider.step = 1;
    slider.value = 50; // center = default value

    slider.addEventListener('input', () => {
      // Higher slider → smaller midpoint → brighter.
      config[key] = midpointFromSlider(parseFloat(slider.value), defaultMidpoint);
    });

    wrapper.appendChild(labelEl);
    wrapper.appendChild(slider);
    container.appendChild(wrapper);
  }
}

function midpointFromSlider(sliderValue, defaultMidpoint) {
  // sliderValue: 0 (dark) → 100 (bright), center 50 = default.
  // midpoint is inverse of brightness, so we negate the exponent.
  return defaultMidpoint * Math.exp(LOG_RANGE * (50 - sliderValue) / 50);
}

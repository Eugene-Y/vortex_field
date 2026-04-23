'use strict';

import { RENDER_DEFAULTS, BRIGHTNESS_SLIDER_POSITIONS, PHYSICS_DEFAULTS, MOUSE_DEFAULTS, ROTATION_FIELD } from '../config/SimulationConfig.js';


const VEL_TONE_BASE     = 30.0;
const ROT_TONE_BASE     = 0.3;
const BRIGHTNESS_LOG_RANGE = 3;

export class ControlPanel {
  constructor(velocityContainer, rotationContainer, physicsContainer, fieldSize, gapSize) {
    velocityContainer.style.width = `${fieldSize}px`;
    rotationContainer.style.width = `${fieldSize}px`;
    velocityContainer.parentElement.style.gap = `${gapSize}px`;

    this._addBrightnessSlider(velocityContainer, 'Brightness', RENDER_DEFAULTS, 'velocityToneMidpoint', VEL_TONE_BASE, BRIGHTNESS_SLIDER_POSITIONS.velocity, 'velocity');
    this._addBrightnessSlider(rotationContainer, 'Brightness', RENDER_DEFAULTS, 'rotationToneMidpoint', ROT_TONE_BASE, BRIGHTNESS_SLIDER_POSITIONS.rotation, 'rotation');

    this._buildPhysicsSliders(physicsContainer);
  }

  addRotationSliders(container) {
    this._addLinearSlider(container, 'Pattern size', 0, 1,
      () => MOUSE_DEFAULTS.patternScale,
      v  => { MOUSE_DEFAULTS.patternScale = v; }
    );
    this._addLinearSlider(container, 'Pair range', 0, 1,
      () => ROTATION_FIELD.pairRange,
      v  => { ROTATION_FIELD.pairRange = v; }
    );
  }

  _buildPhysicsSliders(container) {
    this._addLogSlider(container, 'Brush radius',   0.5,    20,
      () => MOUSE_DEFAULTS.impulseRadius,
      v  => { MOUSE_DEFAULTS.impulseRadius = v; }
    );
    this._addLogSlider(container, 'Brush strength', 1,      500,
      () => MOUSE_DEFAULTS.impulseStrength,
      v  => { MOUSE_DEFAULTS.impulseStrength = v; }
    );
    this._addLogSlider(container, 'Sim speed',      0.1,    10.0,
      () => PHYSICS_DEFAULTS.simulationSpeed,
      v  => { PHYSICS_DEFAULTS.simulationSpeed = v; }
    );
    // Damping slider works in "loss per frame" space (1 - damping) for a clean log scale.
    // Loss 0.0005 ≈ damping 0.9995 (very slow decay) to 0.2 ≈ damping 0.8 (fast decay).
    this._addLogSlider(container, 'Damping loss',   0.0005, 0.2,
      () => 1 - PHYSICS_DEFAULTS.damping,
      v  => { PHYSICS_DEFAULTS.damping = 1 - v; }
    );
  }

  _addBrightnessSlider(container, label, config, key, toneBase, initialSliderValue, positionKey) {
    const applyPosition = sliderValue => {
      config[key] = toneBase * Math.exp(BRIGHTNESS_LOG_RANGE * (50 - sliderValue) / 50);
      BRIGHTNESS_SLIDER_POSITIONS[positionKey] = sliderValue;
    };
    applyPosition(initialSliderValue);
    this._addSlider({
      container,
      label,
      initialSliderValue,
      formatValue: null,
      onChange: applyPosition,
    });
  }

  // Linear slider. getValue/setValue work in the natural value space.
  _addLinearSlider(container, label, min, max, getValue, setValue) {
    const defaultValue       = getValue();
    const initialSliderValue = Math.round((defaultValue - min) / (max - min) * 100);

    this._addSlider({
      container,
      label,
      initialSliderValue: Math.max(0, Math.min(100, initialSliderValue)),
      formatValue: sliderValue => {
        const v = min + (max - min) * sliderValue / 100;
        return v.toFixed(2);
      },
      onChange: sliderValue => {
        setValue(min + (max - min) * sliderValue / 100);
      },
    });
  }

  // Log-scaled slider. getValue/setValue work in the natural value space (what gets stored).
  // Slider center always corresponds to the current default value at construction time.
  _addLogSlider(container, label, min, max, getValue, setValue) {
    const logMin = Math.log(min);
    const logMax = Math.log(max);
    const defaultValue = getValue();
    const initialSliderValue = Math.round(
      (Math.log(defaultValue) - logMin) / (logMax - logMin) * 100
    );

    this._addSlider({
      container,
      label,
      initialSliderValue: Math.max(0, Math.min(100, initialSliderValue)),
      formatValue: sliderValue => {
        const v = Math.exp(logMin + (logMax - logMin) * sliderValue / 100);
        return v < 0.01 ? v.toExponential(1) : v.toPrecision(3);
      },
      onChange: sliderValue => {
        setValue(Math.exp(logMin + (logMax - logMin) * sliderValue / 100));
      },
    });
  }

  _addSlider({ container, label, initialSliderValue, formatValue, onChange }) {
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
    slider.value = initialSliderValue;

    const valueEl = document.createElement('span');
    valueEl.className = 'control-value';
    valueEl.textContent = formatValue ? formatValue(initialSliderValue) : '';

    slider.addEventListener('input', () => {
      const v = parseFloat(slider.value);
      onChange(v);
      if (formatValue) valueEl.textContent = formatValue(v);
    });

    wrapper.appendChild(labelEl);
    wrapper.appendChild(slider);
    if (formatValue) wrapper.appendChild(valueEl);
    container.appendChild(wrapper);
  }
}

'use strict';

import { RENDER_DEFAULTS, BRIGHTNESS_SLIDER_POSITIONS, PHYSICS_DEFAULTS, MOUSE_DEFAULTS, ROTATION_FIELD, VEL_LOG_RANGE, ROT_LOG_RANGE } from '../config/SimulationConfig.js';

const VEL_TONE_BASE = 30.0;
const ROT_TONE_BASE = 0.3;

export class ControlPanel {
  constructor(velocityContainer, rotationContainer, physicsContainer, fieldSize, gapSize) {
    velocityContainer.style.width = `${fieldSize}px`;
    rotationContainer.style.width = `${fieldSize}px`;

    this._addBrightnessSlider(velocityContainer, 'Brightness', RENDER_DEFAULTS, 'velocityToneMidpoint', VEL_TONE_BASE, VEL_LOG_RANGE, BRIGHTNESS_SLIDER_POSITIONS.velocity, 'velocity');
    this._addBrightnessSlider(rotationContainer, 'Brightness', RENDER_DEFAULTS, 'rotationToneMidpoint', ROT_TONE_BASE, ROT_LOG_RANGE, BRIGHTNESS_SLIDER_POSITIONS.rotation, 'rotation');

    this._buildPhysicsSliders(physicsContainer);
  }

  addRotationSliders(container) {
    this._addLinearSlider(container, 'Pattern size', 0, 1,
      () => MOUSE_DEFAULTS.patternScale,
      v => { MOUSE_DEFAULTS.patternScale = v; }
    );
    this._addSymmetricPowerSlider(container, 'Pair range', -1, 1, 0.4,
      () => ROTATION_FIELD.pairRange,
      v => { ROTATION_FIELD.pairRange = v; },
      'Positive: local rotation centers first. Negative: distant pairs first. Higher absolute value → more GPU load.'
    );
  }

  _buildPhysicsSliders(container) {
    const boundaryRow = this._addSelect(container, 'Boundary', [
      { value: 0, label: 'Wrap' },
      { value: 1, label: 'Absorb' },
      { value: 2, label: 'Reflect' },
    ],
      () => PHYSICS_DEFAULTS.boundaryMode,
      v => { PHYSICS_DEFAULTS.boundaryMode = v; }
    );
    boundaryRow.querySelector('select').style.width = '70px';
    boundaryRow.appendChild(this._createGridSizeInput());
    this._addLogSlider(container, 'Brush radius', 0.5, 20,
      () => MOUSE_DEFAULTS.impulseRadius,
      v => { MOUSE_DEFAULTS.impulseRadius = v; }
    );
    this._addLogSlider(container, 'Brush strength', 1, 500,
      () => MOUSE_DEFAULTS.impulseStrength,
      v => { MOUSE_DEFAULTS.impulseStrength = v; }
    );
    this._addLogSlider(container, 'dt', 0.0001, 10.0,
      () => PHYSICS_DEFAULTS.simulationSpeed,
      v => { PHYSICS_DEFAULTS.simulationSpeed = v; }
    );
    // Damping slider works in "loss per frame" space (1 - damping) for a clean log scale.
    // Loss 0.0005 ≈ damping 0.9995 (very slow decay) to 0.2 ≈ damping 0.8 (fast decay).
    this._addLogSlider(container, 'Damping loss', 0.000000001, 0.2,
      () => 1 - PHYSICS_DEFAULTS.damping,
      v => { PHYSICS_DEFAULTS.damping = 1 - v; }
    );
    this._addLinearSlider(container, 'Vorticity', 0, 2,
      () => PHYSICS_DEFAULTS.vorticityStrength,
      v => { PHYSICS_DEFAULTS.vorticityStrength = v; }
    );
    // Slider is inverted: right = gas (few iterations, compressible),
    // left = liquid (many iterations, incompressible).
    this._addIntSlider(container, 'Incompressibility (gas->liquid)', 1, 100,
      () => 101 - PHYSICS_DEFAULTS.pressureIterations,
      v => { PHYSICS_DEFAULTS.pressureIterations = 101 - v; }
    );
  }

  _addBrightnessSlider(container, label, config, key, toneBase, logRange, initialSliderValue, positionKey) {
    const applyPosition = sliderValue => {
      config[key] = toneBase * Math.exp(logRange * (50 - sliderValue) / 50);
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
  // Power-curve slider symmetric around center (maps to mid of [min,max]).
  // Exponent > 1 gives finer control near center; exponent = 1 is linear.
  _addSymmetricPowerSlider(container, label, min, max, exponent, getValue, setValue, hint = null) {
    const mid = (min + max) / 2;
    const halfRange = (max - min) / 2;
    const toSlider = v => Math.sign(v - mid) * Math.pow(Math.abs((v - mid) / halfRange), exponent) * 50 + 50;
    const fromSlider = t => mid + Math.sign(t - 50) * Math.pow(Math.abs((t - 50) / 50), 1 / exponent) * halfRange;
    const initialSliderValue = Math.round(Math.max(0, Math.min(100, toSlider(getValue()))));

    this._addSlider({
      container,
      label,
      initialSliderValue,
      formatValue: sliderValue => fromSlider(sliderValue).toFixed(3),
      onChange: sliderValue => setValue(fromSlider(sliderValue)),
      hint,
    });
  }

  _addIntSlider(container, label, min, max, getValue, setValue) {
    const defaultValue = getValue();
    const initialSliderValue = Math.round((defaultValue - min) / (max - min) * 100);

    this._addSlider({
      container,
      label,
      initialSliderValue: Math.max(0, Math.min(100, initialSliderValue)),
      formatValue: sliderValue => {
        const v = Math.round(min + (max - min) * sliderValue / 100);
        return String(v);
      },
      onChange: sliderValue => {
        setValue(Math.round(min + (max - min) * sliderValue / 100));
      },
    });
  }

  _addLinearSlider(container, label, min, max, getValue, setValue) {
    const defaultValue = getValue();
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

  _createGridSizeInput() {
    const label = document.createElement('span');
    label.className = 'control-label';
    label.textContent = 'Grid size';
    label.style.marginLeft = '12px';

    const input = document.createElement('input');
    input.id = 'grid-size-input';
    input.type = 'number';
    input.min = '32';
    input.max = '512';

    const fragment = document.createDocumentFragment();
    fragment.appendChild(label);
    fragment.appendChild(input);
    return fragment;
  }

  _addSelect(container, label, options, getValue, setValue) {
    const wrapper = document.createElement('div');
    wrapper.className = 'control-row';

    const labelEl = document.createElement('span');
    labelEl.className = 'control-label';
    labelEl.textContent = label;

    const select = document.createElement('select');
    select.className = 'pattern-select';
    const current = getValue();
    for (const { value, label: text } of options) {
      const option = document.createElement('option');
      option.value = value;
      option.textContent = text;
      if (value === current) option.selected = true;
      select.appendChild(option);
    }
    select.addEventListener('input', () => setValue(parseInt(select.value, 10)));

    wrapper.appendChild(labelEl);
    wrapper.appendChild(select);
    container.appendChild(wrapper);
    return wrapper;
  }

  _addSlider({ container, label, initialSliderValue, formatValue, onChange, hint = null }) {
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
    if (hint) {
      const hintEl = document.createElement('div');
      hintEl.className = 'control-hint';
      hintEl.textContent = hint;
      container.appendChild(hintEl);
    }
  }
}

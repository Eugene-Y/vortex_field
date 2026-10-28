'use strict';

import { RENDER_DEFAULTS, BRIGHTNESS_SLIDER_POSITIONS, PHYSICS_DEFAULTS, MOUSE_DEFAULTS, ROTATION_FIELD, VEL_LOG_RANGE, ROT_DIM_LOG_RANGE, ROT_BRIGHT_LOG_RANGE, DENS_LOG_RANGE, GRAVITY, GRAVITY_EXPERIMENT, BRUSH } from '../config/SimulationConfig.js';

const VEL_TONE_BASE  = 30.0;
const ROT_TONE_BASE  = 1.0;
const DENS_TONE_BASE = 10.0;

const VORTICITY_MAX      = 50.0;
const VORTICITY_EXPONENT = 2.0;   // >1 gives finer steps near zero

const PAIR_DISTANCE_SLIDER_STEPS = 1000;

const CONTROL_HINTS = {
  velocityBrightness: 'Field A brightness. Move right to make slow flow visible.',
  massBrightness:     'Brightness of the mass overlay drawn on top of Field A.',
  rotationBrightness: 'Field B brightness. Move right to reveal weak rotation.',
  autoNormalize:      'Auto: divide Field B by its smoothed mean energy, so brightness stays steady as the flow speeds up or calms down.',
  brushMode:          'What strokes and patterns inject: Impulse pushes the fluid, Mass drops matter that gravity pulls on.',
  brushRadius:        'Brush radius in grid cells for strokes and patterns. Also sets the size of the focus circle on Field B.',
  brushStrength:      'How hard strokes and patterns push the fluid. Very high values make the flow chaotic.',
  speedSensitivity:   'How much mouse speed scales stroke strength: 0 = constant, 1 = proportional to speed.',
  boundary:           'Field edges: Wrap joins opposite edges, Absorb lets fluid flow out, Reflect makes them solid walls.',
  gridSize:           'Cells per side (32–1024). Changing it reloads the page; GPU load grows quickly with size.',
  dt:                 'Simulation speed. Below 1 the time step shrinks (slow motion); above 1 several steps run per frame.',
  dampingLoss:        'Share of velocity lost per second. Left: motion lasts almost forever; right: it dies out quickly.',
  vorticity:          'Vorticity confinement: re-energises small swirls that numerical diffusion would smear out. 0 = off.',
  incompressibility:  'Pressure solver effort: more iterations keep the flow incompressible (liquid-like), fewer let it compress (gas-like).',
  gravityEnabled:     'Turn on self-gravity: injected mass attracts the surrounding fluid.',
  gravityStrength:    'How strongly mass pulls on the fluid.',
  gravityIterations:  'Solver iterations for the gravitational potential per step: more is more accurate but slower.',
  massDamping:        'Share of mass kept per second: below 1, injected mass slowly fades.',
  patternSize:        'Size of the pattern injected by double-clicking Field B, as a fraction of the field.',
  showContribution:   'Which rotation directions Field B accumulates: CCW (orange), CW (blue), or both.',
  pairDistance:       'Distance between paired cells, as a fraction of half the grid: small values show local swirls, large ones large-scale rotation.',
  distanceDelta:      'Spread around the pair distance. Wider includes more pairs: smoother image, more GPU work.',
  sampleStride:       'Use every N-th cell in each axis for Field B: N² less work, brightness is compensated.',
};

export class ControlPanel {
  constructor(velocityContainer, rotationContainer, physicsContainer, fieldSize, gapSize) {
    velocityContainer.style.width = `${fieldSize}px`;
    rotationContainer.style.width = `${fieldSize}px`;

    this._addBrightnessSlider(velocityContainer, 'Brightness', RENDER_DEFAULTS, 'velocityToneMidpoint', VEL_TONE_BASE, VEL_LOG_RANGE, VEL_LOG_RANGE, BRIGHTNESS_SLIDER_POSITIONS.velocity, 'velocity', CONTROL_HINTS.velocityBrightness);
    if (GRAVITY_EXPERIMENT) {
      this._addBrightnessSlider(velocityContainer, 'Mass brightness', RENDER_DEFAULTS, 'densityToneMidpoint', DENS_TONE_BASE, DENS_LOG_RANGE, DENS_LOG_RANGE, BRIGHTNESS_SLIDER_POSITIONS.density, 'density', CONTROL_HINTS.massBrightness);
    }
    const rotBrightnessRow = this._addBrightnessSlider(rotationContainer, 'Brightness', RENDER_DEFAULTS, 'rotationToneMidpoint', ROT_TONE_BASE, ROT_DIM_LOG_RANGE, ROT_BRIGHT_LOG_RANGE, BRIGHTNESS_SLIDER_POSITIONS.rotation, 'rotation', CONTROL_HINTS.rotationBrightness);
    this._addAutoNormalizeButton(rotBrightnessRow);

    this._buildPhysicsSliders(physicsContainer);
  }

  // Returns the pattern-size row so PatternInjector can append the inject dropdown to it.
  addRotationSliders(container) {
    const patternRow = this._addLinearSlider(container, 'Pattern size', 0, 1,
      () => MOUSE_DEFAULTS.patternScale,
      v => { MOUSE_DEFAULTS.patternScale = v; },
      CONTROL_HINTS.patternSize
    );
    this._addContributionToggles(container);
    this._addLogSlider(container, 'Pair distance', 0.001, 1.0,
      () => ROTATION_FIELD.pairDistance,
      v => { ROTATION_FIELD.pairDistance = v; },
      CONTROL_HINTS.pairDistance,
      PAIR_DISTANCE_SLIDER_STEPS
    );
    this._addLogSlider(container, 'Distance delta', 0.001, 1.0,
      () => ROTATION_FIELD.distanceDelta,
      v => { ROTATION_FIELD.distanceDelta = v; },
      CONTROL_HINTS.distanceDelta
    );
    this._addSlider({
      container,
      label: 'Sample stride',
      steps: 5,
      initialSliderValue: Math.log2(ROTATION_FIELD.sampleStride),
      formatValue: v => String(2 ** Math.round(v)),
      onChange: v => { ROTATION_FIELD.sampleStride = 2 ** Math.round(v); },
      hint: CONTROL_HINTS.sampleStride,
    });
    return patternRow;
  }

  _buildPhysicsSliders(container) {
    this._addBrushControls(container);
    this._addBoundaryControls(container);
    this._addFlowSliders(container);
    if (GRAVITY_EXPERIMENT) this._addGravityControls(container);
  }

  _addBrushControls(container) {
    if (GRAVITY_EXPERIMENT) this._addBrushModeToggles(container);
    const brushRow = this._addLogSlider(container, 'Brush radius', 0.5, 256,
      () => MOUSE_DEFAULTS.impulseRadius,
      v => { MOUSE_DEFAULTS.impulseRadius = v; },
      CONTROL_HINTS.brushRadius
    );
    this._appendLogSlider(brushRow, 'Strength', 1, 500,
      () => MOUSE_DEFAULTS.impulseStrength,
      v => { MOUSE_DEFAULTS.impulseStrength = v; },
      CONTROL_HINTS.brushStrength
    );
    this._addLinearSlider(container, 'Speed sensitivity', 0, 1,
      () => MOUSE_DEFAULTS.speedSensitivity,
      v => { MOUSE_DEFAULTS.speedSensitivity = v; },
      CONTROL_HINTS.speedSensitivity
    );
  }

  _addBoundaryControls(container) {
    const boundaryRow = this._addSelect(container, 'Boundary', [
      { value: 0, label: 'Wrap' },
      { value: 1, label: 'Absorb' },
      { value: 2, label: 'Reflect' },
    ],
      () => PHYSICS_DEFAULTS.boundaryMode,
      v => { PHYSICS_DEFAULTS.boundaryMode = v; },
      CONTROL_HINTS.boundary
    );
    boundaryRow.querySelector('select').style.width = '70px';
    boundaryRow.appendChild(this._createGridSizeInput(CONTROL_HINTS.gridSize));
  }

  _addFlowSliders(container) {
    this._addLogSlider(container, 'dt', 0.1, 10.0,
      () => PHYSICS_DEFAULTS.simulationSpeed,
      v => { PHYSICS_DEFAULTS.simulationSpeed = v; },
      CONTROL_HINTS.dt
    );
    // Works in loss space (1 − damping) so the log scale can resolve retention values just below 1.
    this._addLogSlider(container, 'Damping loss', 0.000000001, 0.2,
      () => 1 - PHYSICS_DEFAULTS.damping,
      v => { PHYSICS_DEFAULTS.damping = 1 - v; },
      CONTROL_HINTS.dampingLoss
    );
    this._addSlider({
      container,
      label: 'Vorticity',
      steps: 100,
      initialSliderValue: Math.round(Math.pow(PHYSICS_DEFAULTS.vorticityStrength / VORTICITY_MAX, 1 / VORTICITY_EXPONENT) * 100),
      formatValue: t => (VORTICITY_MAX * Math.pow(t / 100, VORTICITY_EXPONENT)).toFixed(2),
      onChange:    t => { PHYSICS_DEFAULTS.vorticityStrength = VORTICITY_MAX * Math.pow(t / 100, VORTICITY_EXPONENT); },
      hint: CONTROL_HINTS.vorticity,
    });
    // Slider is inverted: right = gas (few iterations, compressible),
    // left = liquid (many iterations, incompressible).
    this._addIntSlider(container, 'Incompressibility (gas->liquid)', 1, 100,
      () => 101 - PHYSICS_DEFAULTS.pressureIterations,
      v => { PHYSICS_DEFAULTS.pressureIterations = 101 - v; },
      CONTROL_HINTS.incompressibility
    );
  }

  _addBrushModeToggles(container) {
    const wrapper = document.createElement('div');
    wrapper.className = 'control-row';
    this._setHint(wrapper, CONTROL_HINTS.brushMode);

    const label = document.createElement('span');
    label.className = 'control-label';
    label.textContent = 'Brush mode';
    wrapper.appendChild(label);

    const group = document.createElement('div');
    group.style.cssText = 'display:flex;gap:4px';

    const makeMode = (mode, text) => {
      const btn = this._createToggleButton(text);
      this._showToggleActive(btn, BRUSH.mode === mode);
      btn.addEventListener('click', () => {
        BRUSH.mode = mode;
        this._showToggleActive(impulseBtn, BRUSH.mode === 'impulse');
        this._showToggleActive(massBtn,    BRUSH.mode === 'mass');
        btn.dispatchEvent(new Event('input', { bubbles: true }));
      });
      return btn;
    };

    const impulseBtn = makeMode('impulse', 'Impulse');
    const massBtn    = makeMode('mass',    'Mass');
    group.appendChild(impulseBtn);
    group.appendChild(massBtn);
    wrapper.appendChild(group);
    container.appendChild(wrapper);
  }

  _addGravityControls(container) {
    const wrapper = document.createElement('div');
    wrapper.className = 'control-row';
    this._setHint(wrapper, CONTROL_HINTS.gravityEnabled);

    const label = document.createElement('span');
    label.className = 'control-label';
    label.textContent = 'Gravity';
    wrapper.appendChild(label);

    const btn = this._createToggleButton('Enable');
    this._showToggleActive(btn, GRAVITY.enabled);
    btn.addEventListener('click', () => {
      GRAVITY.enabled = !GRAVITY.enabled;
      this._showToggleActive(btn, GRAVITY.enabled);
      btn.dispatchEvent(new Event('input', { bubbles: true }));
    });
    wrapper.appendChild(btn);
    container.appendChild(wrapper);

    this._addLogSlider(container, 'Gravity strength', 0.01, 500,
      () => GRAVITY.strength,
      v => { GRAVITY.strength = v; },
      CONTROL_HINTS.gravityStrength
    );
    this._addIntSlider(container, 'Gravity iterations', 1, 80,
      () => GRAVITY.iterations,
      v => { GRAVITY.iterations = v; },
      CONTROL_HINTS.gravityIterations
    );
    this._addLogSlider(container, 'Mass damping', 0.9, 1.0,
      () => GRAVITY.densityDamping,
      v => { GRAVITY.densityDamping = v; },
      CONTROL_HINTS.massDamping
    );
  }

  _addBrightnessSlider(container, label, config, key, toneBase, dimLogRange, brightLogRange, initialSliderValue, positionKey, hint) {
    const applyPosition = sliderValue => {
      const t = (50 - sliderValue) / 50;
      config[key] = toneBase * Math.exp((t >= 0 ? dimLogRange : brightLogRange) * t);
      BRIGHTNESS_SLIDER_POSITIONS[positionKey] = sliderValue;
    };
    applyPosition(initialSliderValue);
    return this._addSlider({
      container,
      label,
      initialSliderValue,
      formatValue: null,
      onChange: applyPosition,
      hint,
    });
  }

  _addAutoNormalizeButton(row) {
    const btn = this._createToggleButton('Auto');
    this._setHint(btn, CONTROL_HINTS.autoNormalize);
    this._showToggleActive(btn, ROTATION_FIELD.autoNormalize);
    btn.addEventListener('click', () => {
      ROTATION_FIELD.autoNormalize ^= 1;
      this._showToggleActive(btn, ROTATION_FIELD.autoNormalize);
      btn.dispatchEvent(new Event('input', { bubbles: true }));
    });
    row.appendChild(btn);
  }

  // Power-curve slider symmetric around center (maps to mid of [min,max]).
  // Exponent > 1 gives finer control near center; exponent = 1 is linear.
  _addSymmetricPowerSlider(container, label, min, max, exponent, getValue, setValue, hint = null, steps = 100, decimals = 3) {
    const mid = (min + max) / 2;
    const halfRange = (max - min) / 2;
    const half = steps / 2;
    const toSlider = v => Math.sign(v - mid) * Math.pow(Math.abs((v - mid) / halfRange), exponent) * half + half;
    const fromSlider = t => mid + Math.sign(t - half) * Math.pow(Math.abs((t - half) / half), 1 / exponent) * halfRange;
    const initialSliderValue = Math.round(Math.max(0, Math.min(steps, toSlider(getValue()))));

    this._addSlider({
      container,
      label,
      initialSliderValue,
      steps,
      formatValue: sliderValue => fromSlider(sliderValue).toFixed(decimals),
      onChange: sliderValue => setValue(fromSlider(sliderValue)),
      hint,
    });
  }

  _addIntSlider(container, label, min, max, getValue, setValue, hint) {
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
      hint,
    });
  }

  _addLinearSlider(container, label, min, max, getValue, setValue, hint) {
    const defaultValue = getValue();
    const initialSliderValue = Math.round((defaultValue - min) / (max - min) * 100);

    return this._addSlider({
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
      hint,
    });
  }

  // Log-scaled slider. getValue/setValue work in the natural value space (what gets stored).
  // Slider center always corresponds to the current default value at construction time.
  _addLogSlider(container, label, min, max, getValue, setValue, hint, steps = 100) {
    return this._addSlider({
      container,
      label,
      hint,
      ...this._logSliderOptions(min, max, getValue, setValue, steps),
    });
  }

  // Use when two sliders share one control-row, e.g. brush radius + strength.
  _appendLogSlider(row, label, min, max, getValue, setValue, hint) {
    row.appendChild(this._createSliderBox({
      label,
      hint,
      ...this._logSliderOptions(min, max, getValue, setValue),
    }));
  }

  _logSliderOptions(min, max, getValue, setValue, steps = 100) {
    const logMin = Math.log(min);
    const logMax = Math.log(max);
    const sliderValueToNatural = sliderValue => Math.exp(logMin + (logMax - logMin) * sliderValue / steps);
    const initialSliderValue = Math.max(0, Math.min(steps, Math.round(
      (Math.log(getValue()) - logMin) / (logMax - logMin) * steps
    )));
    const formatValue = sliderValue => {
      const v = sliderValueToNatural(sliderValue);
      return v < 0.001 ? v.toExponential(1) : v.toPrecision(3);
    };
    return {
      initialSliderValue,
      steps,
      formatValue,
      onChange: sliderValue => { setValue(sliderValueToNatural(sliderValue)); },
    };
  }

  // The group is display:contents, so label and input still lay out as direct children of the row.
  _createGridSizeInput(hint) {
    const group = document.createElement('span');
    group.className = 'hint-group';
    this._setHint(group, hint);

    const label = document.createElement('span');
    label.className = 'control-label';
    label.textContent = 'Grid size';
    label.style.marginLeft = '12px';

    const input = document.createElement('input');
    input.id = 'grid-size-input';
    input.type = 'number';
    input.min = '32';
    input.max = '1024';

    group.appendChild(label);
    group.appendChild(input);
    return group;
  }

  _addContributionToggles(container) {
    const wrapper = document.createElement('div');
    wrapper.className = 'control-row';
    this._setHint(wrapper, CONTROL_HINTS.showContribution);

    const label = document.createElement('span');
    label.className = 'control-label';
    label.textContent = 'Show contribution';
    wrapper.appendChild(label);

    const group = document.createElement('div');
    group.style.cssText = 'display:flex;gap:4px';

    const makeBit = (bit, text) => {
      const btn = this._createToggleButton(text);
      const active = () => !!(ROTATION_FIELD.showMask & bit);
      this._showToggleActive(btn, active());
      btn.addEventListener('click', () => {
        ROTATION_FIELD.showMask ^= bit;
        this._showToggleActive(btn, active());
        btn.dispatchEvent(new Event('input', { bubbles: true }));
      });
      return btn;
    };

    group.appendChild(makeBit(1, 'CCW'));
    group.appendChild(makeBit(2, 'CW'));
    wrapper.appendChild(group);
    container.appendChild(wrapper);
  }

  _createToggleButton(text) {
    const btn = document.createElement('button');
    btn.className = 'toggle-button';
    btn.textContent = text;
    return btn;
  }

  _showToggleActive(btn, active) {
    btn.classList.toggle('active', Boolean(active));
  }

  // ControlHintDisplay shows the nearest data-hint of whatever the pointer or focus is on.
  _setHint(element, hint) {
    if (hint) element.dataset.hint = hint;
  }

  _addSelect(container, label, options, getValue, setValue, hint) {
    const wrapper = document.createElement('div');
    wrapper.className = 'control-row';
    this._setHint(wrapper, hint);

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

  _addSlider({ container, label, initialSliderValue, formatValue, onChange, hint = null, steps = 100 }) {
    const wrapper = document.createElement('div');
    wrapper.className = 'control-row';
    wrapper.appendChild(this._createSliderBox({ label, initialSliderValue, formatValue, onChange, hint, steps }));
    container.appendChild(wrapper);
    return wrapper;
  }

  // A rectangle whose whole area is the range input; label and value are drawn on top of it.
  _createSliderBox({ label, initialSliderValue, formatValue, onChange, hint = null, steps = 100 }) {
    const box = document.createElement('div');
    box.className = 'slider-box';
    this._setHint(box, hint);

    const slider = document.createElement('input');
    slider.type = 'range';
    slider.min = 0;
    slider.max = steps;
    slider.step = 1;
    slider.value = initialSliderValue;
    slider.setAttribute('aria-label', label);

    const labelEl = document.createElement('span');
    labelEl.className = 'slider-label';
    labelEl.textContent = label;

    const valueEl = document.createElement('span');
    valueEl.className = 'slider-value';

    const showSliderValue = sliderValue => {
      box.style.setProperty('--fill', `${sliderValue / steps * 100}%`);
      if (formatValue) valueEl.textContent = formatValue(sliderValue);
    };
    showSliderValue(initialSliderValue);

    slider.addEventListener('input', () => {
      const sliderValue = parseFloat(slider.value);
      onChange(sliderValue);
      showSliderValue(sliderValue);
    });

    box.appendChild(slider);
    box.appendChild(labelEl);
    if (formatValue) box.appendChild(valueEl);
    return box;
  }
}

'use strict';

const CONCEALED_CLASS = 'description-concealed';

// Swaps the page description for the data-hint of the control under the pointer or keyboard focus.
// Both texts stay in the layout (only one is visible), so the page never jumps when they swap.
export class ControlHintDisplay {
  constructor(descriptionElement, hintElement) {
    this._descriptionElement = descriptionElement;
    this._hintElement        = hintElement;

    document.addEventListener('pointerover', this._onPointerOver.bind(this));
    document.documentElement.addEventListener('pointerleave', () => this._showDescription());
    document.addEventListener('focusin', event => this._showHintFor(event.target));
    document.addEventListener('focusout', () => this._showDescription());
  }

  _onPointerOver(event) {
    // While a slider is being dragged the pointer often slides off it; keep its hint until release.
    if (event.buttons !== 0) return;
    this._showHintFor(event.target);
  }

  _showHintFor(element) {
    const hint = element.closest('[data-hint]')?.dataset.hint;
    if (hint) {
      this._showHint(hint);
    } else {
      this._showDescription();
    }
  }

  _showHint(text) {
    this._hintElement.textContent = text;
    this._hintElement.classList.remove(CONCEALED_CLASS);
    this._descriptionElement.classList.add(CONCEALED_CLASS);
  }

  _showDescription() {
    this._hintElement.classList.add(CONCEALED_CLASS);
    this._descriptionElement.classList.remove(CONCEALED_CLASS);
  }
}

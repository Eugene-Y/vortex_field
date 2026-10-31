'use strict';

const SPACE_KEY = ' ';

const isTypeAheadKey = event =>
  event.key.length === 1 && event.key !== SPACE_KEY && !event.metaKey && !event.ctrlKey && !event.altKey;

// A focused dropdown jumps to the first option starting with a typed letter, so after picking a
// pattern, P or R would silently change it. The key still reaches the page shortcuts; arrows,
// Space and Enter still operate the dropdown.
export function blockSelectTypeAhead() {
  document.addEventListener('keydown', event => {
    if (event.target.closest?.('select') && isTypeAheadKey(event)) event.preventDefault();
  }, { capture: true });
}

'use strict';

// Physical key, so the shortcut works whatever keyboard layout is active.
const RANDOM_SCENE_KEY_CODE = 'KeyR';

/** Calls startRandomScene when R is pressed; Cmd/Ctrl+R still reloads the page. */
export class RandomSceneShortcut {
  constructor(startRandomScene) {
    this._startRandomScene = startRandomScene;
    this._onKeyDown        = this._onKeyDown.bind(this);
    window.addEventListener('keydown', this._onKeyDown);
  }

  dispose() {
    window.removeEventListener('keydown', this._onKeyDown);
  }

  _onKeyDown(event) {
    if (event.code !== RANDOM_SCENE_KEY_CODE || event.repeat) return;
    if (event.metaKey || event.ctrlKey || event.altKey) return;
    if (this._isTypingIntoField(event.target)) return;
    this._startRandomScene();
  }

  _isTypingIntoField(target) {
    return target.matches?.('input[type=number], input[type=text], textarea, [contenteditable]') ?? false;
  }
}

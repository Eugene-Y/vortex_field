'use strict';

import { createPhysicsStep } from './PhysicsRegistry.js';
import { DEFAULT_PHYSICS_MODEL } from '../config/SimulationConfig.js';

/**
 * Field A: the velocity field.
 * Owns the physics step object and delegates all simulation logic to it.
 * The physics model is injected at construction time — swapping models
 * requires only changing the modelId, not this class.
 */
export class FluidField {
  constructor(device, canvas, shaderSources, modelId = DEFAULT_PHYSICS_MODEL) {
    this._physicsStep    = createPhysicsStep(modelId, device, canvas, shaderSources);
    this._stepGeneration = 0;
  }

  get velocityTexture() {
    return this._physicsStep.velocityTexture;
  }

  // Increments every step() — lets consumers detect whether velocity changed
  // without relying on texture reference identity (ping-pong swap count may be even).
  get stepGeneration() {
    return this._stepGeneration;
  }

  /** Advances the simulation by deltaTime seconds. */
  step(deltaTime) {
    this._physicsStep.step(deltaTime);
    this._stepGeneration++;
  }

  /** Renders the current velocity field to the canvas. */
  render() {
    this._physicsStep.renderToCanvas();
  }

  injectImpulse(position, direction, radius, strength) {
    this._physicsStep.injectImpulse(position, direction, radius, strength);
  }

  injectDisk(center, radiusUv, strength, mode) {
    this._physicsStep.injectDisk(center, radiusUv, strength, mode);
  }

  addNoise(strength, seed) {
    this._physicsStep.injectNoise(strength, seed);
  }

  reset() {
    this._physicsStep.clearVelocity();
  }

  dispose() {
    this._physicsStep.dispose();
  }
}

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
  constructor(gl, shaderSources, modelId = DEFAULT_PHYSICS_MODEL) {
    this._physicsStep = createPhysicsStep(modelId, gl, shaderSources);
  }

  get velocityTexture() {
    return this._physicsStep.velocityTexture;
  }

  step(deltaTime, quadVao) {
    this._physicsStep.step(deltaTime, quadVao);
  }

  injectImpulse(position, direction, radius, strength, quadVao) {
    this._physicsStep.injectImpulse(position, direction, radius, strength, quadVao);
  }

  injectDisk(center, radiusUv, strength, mode, quadVao) {
    this._physicsStep.injectDisk(center, radiusUv, strength, mode, quadVao);
  }

  addNoise(strength, seed, quadVao) {
    this._physicsStep.injectNoise(strength, seed, quadVao);
  }

  reset() {
    this._physicsStep.clearVelocity();
  }

  dispose() {
    this._physicsStep.dispose();
  }
}

'use strict';

import { PHYSICS_MODELS } from '../config/SimulationConfig.js';
import { NavierStokesStep } from './NavierStokesStep.js';

/**
 * Registry of available physics step implementations.
 * To add a new model: import it here and add an entry below.
 * The rest of the system is unchanged.
 */
const REGISTRY = {
  [PHYSICS_MODELS.NAVIER_STOKES]: NavierStokesStep,
};

/**
 * Instantiates the physics step for the given model identifier.
 * All models receive the same constructor arguments: (gl, shaderSources).
 */
export function createPhysicsStep(modelId, gl, shaderSources) {
  const PhysicsStepClass = REGISTRY[modelId];
  if (!PhysicsStepClass) {
    throw new Error(`Unknown physics model: "${modelId}". Registered models: ${Object.keys(REGISTRY).join(', ')}`);
  }
  return new PhysicsStepClass(gl, shaderSources);
}

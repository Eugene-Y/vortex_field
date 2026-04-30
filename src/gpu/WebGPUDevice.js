'use strict';

/**
 * Initializes a WebGPU adapter and device.
 * Throws a descriptive error if WebGPU is unavailable or the required features
 * are missing — caller should catch and show a user-facing message.
 */
export async function createWebGPUDevice() {
  if (!navigator.gpu) {
    throw new Error('WebGPU is not supported in this browser. Use Chrome 113+ or Safari 17+.');
  }

  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) {
    throw new Error('No WebGPU adapter found. Your GPU or driver may not support WebGPU.');
  }

  const device = await adapter.requestDevice();

  device.lost.then(info => {
    console.error('WebGPU device lost:', info.message);
  });

  return device;
}

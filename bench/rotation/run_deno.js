'use strict';

// deno run --unstable-webgpu --allow-read bench/rotation/run_deno.js [configs] [scenarios] [variants]
// Each argument is a comma-separated key list; '-' keeps the default.

import { runRotationBenchmark } from './RotationBenchmark.js';
import { CONFIGS, VARIANTS, DEFAULT_VARIANT_KEYS, SUBGROUP_VARIANT_KEYS } from './BenchmarkPlan.js';

const repoRoot = new URL('../../', import.meta.url);
const readRepoText = path => Deno.readTextFile(new URL(path, repoRoot));

function parseKeyList(argument, fallback) {
  return !argument || argument === '-' ? fallback : argument.split(',');
}

const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
const timestampSupported = adapter.features.has('timestamp-query');
const subgroupsSupported = adapter.features.has('subgroups');
const requiredFeatures = [];
if (timestampSupported) requiredFeatures.push('timestamp-query');
if (subgroupsSupported) requiredFeatures.push('subgroups');
const device = await adapter.requestDevice({
  requiredFeatures,
  requiredLimits: { maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize, maxBufferSize: adapter.limits.maxBufferSize },
});
device.addEventListener('uncapturederror', event => console.error('GPU error:', event.error.message));

const configKeys   = parseKeyList(Deno.args[0], ['appDefault', 'medium']);
const scenarioKeys = parseKeyList(Deno.args[1], ['vortices', 'solidBody', 'whiteNoise']);
const defaultVariants = subgroupsSupported ? [...DEFAULT_VARIANT_KEYS, ...SUBGROUP_VARIANT_KEYS] : DEFAULT_VARIANT_KEYS;
const variantKeys  = parseKeyList(Deno.args[2], defaultVariants);

console.log(`adapter: ${adapter.info?.description} · timestamps ${timestampSupported} · subgroups ${subgroupsSupported}`);

await runRotationBenchmark({
  device,
  productionSources: {
    compute: await readRepoText('src/shaders/rotation_compute.wgsl'),
    reduce:  await readRepoText('src/shaders/rotation_reduce.wgsl'),
  },
  configs:   configKeys.map(key => CONFIGS[key]),
  scenarios: scenarioKeys,
  variants:  variantKeys.map(key => VARIANTS[key]),
  log:       line => console.log(line),
  timestampSupported,
  loadFieldFile: async fileName => (await Deno.readFile(new URL(`fields/${fileName}`, import.meta.url))).buffer,
});

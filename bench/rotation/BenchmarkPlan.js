'use strict';

import { PRODUCTION_VARIANT } from './RotationBenchmark.js';

const FIXED32_SCALE = 65536;
const FIXED64_SCALE = 4294967296;

export const CONFIGS = {
  appDefault:  { name: 'appDefault',  gridSize: 256, pairDistance: 0.0447, distanceDelta: 0.00302 },
  appDefaultAbsorb: { name: 'appDefaultAbsorb', gridSize: 256, pairDistance: 0.0447, distanceDelta: 0.00302, boundaryMode: 1 },
  mediumAbsorb:     { name: 'mediumAbsorb', gridSize: 256, pairDistance: 0.1, distanceDelta: 0.05, boundaryMode: 1 },
  medium:      { name: 'medium',      gridSize: 256, pairDistance: 0.1,    distanceDelta: 0.05 },
  large:       { name: 'large',       gridSize: 256, pairDistance: 0.3,    distanceDelta: 0.1 },
  grid512:     { name: 'grid512',     gridSize: 512, pairDistance: 0.05,   distanceDelta: 0.025 },
  grid1024:    { name: 'grid1024',    gridSize: 1024, pairDistance: 0.0447, distanceDelta: 0.00302 },
  maskSmall:   { name: 'maskSmall',   gridSize: 256, pairDistance: 0.1, distanceDelta: 0.05,
                 mask: { centerX: 128, centerY: 128, radius: 24 } },
  maskLarge:   { name: 'maskLarge',   gridSize: 512, pairDistance: 0.1, distanceDelta: 0.05,
                 mask: { centerX: 256, centerY: 256, radius: 64 } },
};

const variant = (name, options) => ({
  name,
  pairMath: 'reference', offsetTable: 'full', accumulation: 'casF32', aggregation: 'none',
  accumulationBuffers: 32, offsetSplit: 1, fixedPointScale: FIXED64_SCALE,
  ...options,
});

export const VARIANTS = {
  production:              PRODUCTION_VARIANT,
  branch:                  variant('branch src shader', { branchShader: true, offsetTable: 'rowSpans', accumulation: 'fixed64', accumulationBuffers: 1, cellOrder: 'tiled8', maskOffsetSplit: 16 }),
  generatedReference:      variant('generated reference', {}),
  halfPlane:               variant('halfPlane', { offsetTable: 'halfPlane' }),
  inverseT:                variant('inverseT', { pairMath: 'inverseT' }),
  inverseTExact:           variant('inverseTExact', { pairMath: 'inverseTExact' }),
  hpExactFixed64K1:        variant('hp+invTExact fixed64 K1', { offsetTable: 'halfPlane', pairMath: 'inverseTExact', accumulation: 'fixed64', accumulationBuffers: 1 }),
  hpExactCasK1:            variant('hp+invTExact casF32 K1', { offsetTable: 'halfPlane', pairMath: 'inverseTExact', accumulationBuffers: 1 }),
  halfPlaneInverseT:       variant('halfPlane+inverseT', { offsetTable: 'halfPlane', pairMath: 'inverseT' }),
  loopOnly:                variant('[diag] no accumulation', { offsetTable: 'halfPlane', pairMath: 'inverseT', accumulation: 'none', accumulationBuffers: 1 }),
  casK1:                   variant('hp+invT casF32 K1', { offsetTable: 'halfPlane', pairMath: 'inverseT', accumulationBuffers: 1 }),
  casK4:                   variant('hp+invT casF32 K4', { offsetTable: 'halfPlane', pairMath: 'inverseT', accumulationBuffers: 4 }),
  fixed32K32:              variant('[diag] hp+invT fixed32 K32', { offsetTable: 'halfPlane', pairMath: 'inverseT', accumulation: 'fixed32', fixedPointScale: FIXED32_SCALE }),
  fixed32K1:               variant('[diag] hp+invT fixed32 K1', { offsetTable: 'halfPlane', pairMath: 'inverseT', accumulation: 'fixed32', fixedPointScale: FIXED32_SCALE, accumulationBuffers: 1 }),
  fixed64K32:              variant('hp+invT fixed64 K32', { offsetTable: 'halfPlane', pairMath: 'inverseT', accumulation: 'fixed64' }),
  fixed64K4:               variant('hp+invT fixed64 K4', { offsetTable: 'halfPlane', pairMath: 'inverseT', accumulation: 'fixed64', accumulationBuffers: 4 }),
  fixed64K1:               variant('hp+invT fixed64 K1', { offsetTable: 'halfPlane', pairMath: 'inverseT', accumulation: 'fixed64', accumulationBuffers: 1 }),
  subgroupCasK32:          variant('hp+invT subgroup casF32 K32', { offsetTable: 'halfPlane', pairMath: 'inverseT', aggregation: 'subgroup' }),
  subgroupCasK4:           variant('hp+invT subgroup casF32 K4', { offsetTable: 'halfPlane', pairMath: 'inverseT', aggregation: 'subgroup', accumulationBuffers: 4 }),
  subgroupCasK1:           variant('hp+invT subgroup casF32 K1', { offsetTable: 'halfPlane', pairMath: 'inverseT', aggregation: 'subgroup', accumulationBuffers: 1 }),
  subgroupTiledFixed64K1:  variant('hp+invTExact subgroup fixed64 K1 tiled8', { offsetTable: 'halfPlane', pairMath: 'inverseTExact', aggregation: 'subgroup', accumulation: 'fixed64', accumulationBuffers: 1, cellOrder: 'tiled8' }),
  subgroupFixed64K1:       variant('hp+invT subgroup fixed64 K1', { offsetTable: 'halfPlane', pairMath: 'inverseT', aggregation: 'subgroup', accumulation: 'fixed64', accumulationBuffers: 1 }),
  subgroup2RoundsCasK4:    variant('hp+invT subgroup×2 casF32 K4', { offsetTable: 'halfPlane', pairMath: 'inverseT', aggregation: 'subgroup', aggregationRounds: 2, accumulationBuffers: 4 }),
  tiledFixed64K1:          variant('hp+invTExact fixed64 K1 tiled8', { offsetTable: 'halfPlane', pairMath: 'inverseTExact', accumulation: 'fixed64', accumulationBuffers: 1, cellOrder: 'tiled8' }),
  tiledCasK1:              variant('hp+invTExact casF32 K1 tiled8', { offsetTable: 'halfPlane', pairMath: 'inverseTExact', accumulationBuffers: 1, cellOrder: 'tiled8' }),
  workgroupCache:          variant('hp+invTExact wgCache512 K1', { offsetTable: 'halfPlane', pairMath: 'inverseTExact', aggregation: 'workgroup', accumulationBuffers: 1 }),
  workgroupCacheTiled:     variant('hp+invTExact wgCache512 K1 tiled8', { offsetTable: 'halfPlane', pairMath: 'inverseTExact', aggregation: 'workgroup', accumulationBuffers: 1, cellOrder: 'tiled8' }),
  workgroupCache2048:      variant('hp+invTExact wgCache2048 K1', { offsetTable: 'halfPlane', pairMath: 'inverseTExact', aggregation: 'workgroup', accumulationBuffers: 1, cacheSlots: 2048 }),
  fixed64Cheap:            variant('hp+invTExact fixed64 K1 cheapWrap', { offsetTable: 'halfPlane', pairMath: 'inverseTExact', accumulation: 'fixed64', accumulationBuffers: 1, cheapWrap: true }),
  wgCache256x256:          variant('wgCache256 wg256 tiled cheapWrap', { offsetTable: 'halfPlane', pairMath: 'inverseTExact', aggregation: 'workgroup', accumulationBuffers: 1, cellOrder: 'tiled8', cacheSlots: 256, workgroupSize: 256, cheapWrap: true }),
  wgCache512x256:          variant('wgCache512 wg256 tiled cheapWrap', { offsetTable: 'halfPlane', pairMath: 'inverseTExact', aggregation: 'workgroup', accumulationBuffers: 1, cellOrder: 'tiled8', cacheSlots: 512, workgroupSize: 256, cheapWrap: true }),
  wgCache128x64:           variant('wgCache128 wg64 tiled cheapWrap', { offsetTable: 'halfPlane', pairMath: 'inverseTExact', aggregation: 'workgroup', accumulationBuffers: 1, cellOrder: 'tiled8', cacheSlots: 128, workgroupSize: 64, cheapWrap: true }),
  wgCache256x64:           variant('wgCache256 wg64 tiled cheapWrap', { offsetTable: 'halfPlane', pairMath: 'inverseTExact', aggregation: 'workgroup', accumulationBuffers: 1, cellOrder: 'tiled8', cacheSlots: 256, workgroupSize: 64, cheapWrap: true }),
  wgCache1024x256:         variant('wgCache1024 wg256 tiled cheapWrap', { offsetTable: 'halfPlane', pairMath: 'inverseTExact', aggregation: 'workgroup', accumulationBuffers: 1, cellOrder: 'tiled8', cacheSlots: 1024, workgroupSize: 256, cheapWrap: true }),
  loopOnlyCheap:           variant('[diag] no accumulation cheapWrap', { offsetTable: 'halfPlane', pairMath: 'inverseTExact', accumulation: 'none', accumulationBuffers: 1, cheapWrap: true }),
  loopOnlySpans:           variant('[diag] no accumulation spans', { offsetTable: 'rowSpans', pairMath: 'inverseTExact', accumulation: 'none', accumulationBuffers: 1, cheapWrap: true }),
  fixed64Spans:            variant('fixed64 K1 spans', { offsetTable: 'rowSpans', pairMath: 'inverseTExact', accumulation: 'fixed64', accumulationBuffers: 1, cheapWrap: true }),
  fixed64Storage:          variant('fixed64 K1 cheapWrap storageVel', { offsetTable: 'halfPlane', pairMath: 'inverseTExact', accumulation: 'fixed64', accumulationBuffers: 1, cheapWrap: true, velocitySource: 'storage' }),
  wgCacheSpans:            variant('wgCache256 wg64 tiled spans', { offsetTable: 'rowSpans', pairMath: 'inverseTExact', aggregation: 'workgroup', accumulationBuffers: 1, cellOrder: 'tiled8', cacheSlots: 256, workgroupSize: 64, cheapWrap: true }),
  wgCacheSpansStorage:     variant('wgCache256 wg64 tiled spans storageVel', { offsetTable: 'rowSpans', pairMath: 'inverseTExact', aggregation: 'workgroup', accumulationBuffers: 1, cellOrder: 'tiled8', cacheSlots: 256, workgroupSize: 64, cheapWrap: true, velocitySource: 'storage' }),
  subgroupSpansFixed64:    variant('subgroup fixed64 K1 spans', { offsetTable: 'rowSpans', pairMath: 'inverseTExact', aggregation: 'subgroup', accumulation: 'fixed64', accumulationBuffers: 1, cheapWrap: true }),
  subgroupSpansFixed64Tiled: variant('subgroup fixed64 K1 spans tiled8', { offsetTable: 'rowSpans', pairMath: 'inverseTExact', aggregation: 'subgroup', accumulation: 'fixed64', accumulationBuffers: 1, cheapWrap: true, cellOrder: 'tiled8' }),
  subgroupSpansFixed64x2:  variant('subgroup×2 fixed64 K1 spans', { offsetTable: 'rowSpans', pairMath: 'inverseTExact', aggregation: 'subgroup', aggregationRounds: 2, accumulation: 'fixed64', accumulationBuffers: 1, cheapWrap: true }),
  wgCacheSpans256:         variant('wgCache256 wg256 tiled spans', { offsetTable: 'rowSpans', pairMath: 'inverseTExact', aggregation: 'workgroup', accumulationBuffers: 1, cellOrder: 'tiled8', cacheSlots: 256, workgroupSize: 256, cheapWrap: true }),
  wgCacheSpansSplit4:      variant('wgCache256 wg64 tiled spans split4', { offsetTable: 'rowSpans', pairMath: 'inverseTExact', aggregation: 'workgroup', accumulationBuffers: 1, cellOrder: 'tiled8', cacheSlots: 256, workgroupSize: 64, cheapWrap: true, offsetSplit: 4 }),
  fixed64SpansSplit4:      variant('fixed64 K1 spans split4', { offsetTable: 'rowSpans', pairMath: 'inverseTExact', accumulation: 'fixed64', accumulationBuffers: 1, cheapWrap: true, offsetSplit: 4 }),
  fixed64SpansSplit16:     variant('fixed64 K1 spans split16', { offsetTable: 'rowSpans', pairMath: 'inverseTExact', accumulation: 'fixed64', accumulationBuffers: 1, cheapWrap: true, offsetSplit: 16 }),
  wgCacheSpansSplit16:     variant('wgCache256 wg64 tiled spans split16', { offsetTable: 'rowSpans', pairMath: 'inverseTExact', aggregation: 'workgroup', accumulationBuffers: 1, cellOrder: 'tiled8', cacheSlots: 256, workgroupSize: 64, cheapWrap: true, offsetSplit: 16 }),
  wgCache64Spans:          variant('wgCache64 wg64 tiled spans', { offsetTable: 'rowSpans', pairMath: 'inverseTExact', aggregation: 'workgroup', accumulationBuffers: 1, cellOrder: 'tiled8', cacheSlots: 64, workgroupSize: 64, cheapWrap: true }),
  wgCache32Spans:          variant('wgCache32 wg64 tiled spans', { offsetTable: 'rowSpans', pairMath: 'inverseTExact', aggregation: 'workgroup', accumulationBuffers: 1, cellOrder: 'tiled8', cacheSlots: 32, workgroupSize: 64, cheapWrap: true }),
  split8:                  variant('hp+invT casF32 K32 split8', { offsetTable: 'halfPlane', pairMath: 'inverseT', offsetSplit: 8 }),
};

export const DEFAULT_VARIANT_KEYS = Object.keys(VARIANTS).filter(key => !key.startsWith('subgroup'));
export const SUBGROUP_VARIANT_KEYS = Object.keys(VARIANTS).filter(key => key.startsWith('subgroup'));

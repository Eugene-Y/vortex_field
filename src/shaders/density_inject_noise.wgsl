// Adds random density: a random amount per cell, or Gaussian blobs of random mass on a
// lattice. Used by the noise pattern in mass mode. Lattice helpers and NoiseParams come from
// noise_lattice.wgsl.

@group(0) @binding(0) var densityIn:  texture_2d<f32>;
@group(0) @binding(1) var densityOut: texture_storage_2d<r32float, write>;
@group(0) @binding(2) var<uniform> params: NoiseParams;

fn latticeBlobDensity(cellCenter: vec2f) -> f32 {
  let columns = nodeRangeReaching(cellCenter.x);
  let rows    = nodeRangeReaching(cellCenter.y);
  var density = 0.0;
  for (var row = rows.first; row <= rows.last; row++) {
    let offsetY = offsetToNode(cellCenter.y, row);
    for (var column = columns.first; column <= columns.last; column++) {
      let offset = vec2f(offsetToNode(cellCenter.x, column), offsetY);
      density += randomUnitAtNode(vec2u(nodeId(column), nodeId(row))) * blobWeight(offset);
    }
  }
  return density;
}

fn noiseDensity(cell: vec2i) -> f32 {
  if (usesPerCellNoise()) { return randomUnitAtNode(vec2u(cell)) * params.strength; }
  return latticeBlobDensity(vec2f(cell) + 0.5) * params.strength;
}

@compute @workgroup_size(8, 8)
fn injectDensityNoise(@builtin(global_invocation_id) id: vec3u) {
  let cell = vec2i(id.xy);
  let n    = i32(params.gridSize);
  if (cell.x >= n || cell.y >= n) { return; }

  let existing = textureLoad(densityIn, cell, 0).r;
  let added    = select(noiseDensity(cell), 0.0, isAbsorbGhostCell(cell));
  textureStore(densityOut, cell, vec4f(max(0.0, existing + added), 0.0, 0.0, 0.0));
}

// Adds random velocity: a random direction per cell, or Gaussian blobs with random
// directions on a lattice. Lattice helpers and NoiseParams come from noise_lattice.wgsl.

@group(0) @binding(0) var velocityIn:  texture_2d<f32>;
@group(0) @binding(1) var velocityOut: texture_storage_2d<rg32float, write>;
@group(0) @binding(2) var<uniform> params: NoiseParams;

const TAU = 6.28318530718;

fn randomDirectionAtNode(node: vec2u) -> vec2f {
  let angle = randomUnitAtNode(node) * TAU;
  return vec2f(cos(angle), sin(angle));
}

fn latticeBlobVelocity(cellCenter: vec2f) -> vec2f {
  let columns = nodeRangeReaching(cellCenter.x);
  let rows    = nodeRangeReaching(cellCenter.y);
  var velocity = vec2f(0.0);
  for (var row = rows.first; row <= rows.last; row++) {
    let offsetY = offsetToNode(cellCenter.y, row);
    for (var column = columns.first; column <= columns.last; column++) {
      let offset = vec2f(offsetToNode(cellCenter.x, column), offsetY);
      velocity += randomDirectionAtNode(vec2u(nodeId(column), nodeId(row))) * blobWeight(offset);
    }
  }
  return velocity;
}

fn noiseVelocity(cell: vec2i) -> vec2f {
  if (usesPerCellNoise()) { return randomDirectionAtNode(vec2u(cell)) * params.strength; }
  return latticeBlobVelocity(vec2f(cell) + 0.5) * params.strength;
}

@compute @workgroup_size(8, 8)
fn injectNoise(@builtin(global_invocation_id) id: vec3u) {
  let cell = vec2i(id.xy);
  let n    = i32(params.gridSize);
  if (cell.x >= n || cell.y >= n) { return; }

  let existing = textureLoad(velocityIn, cell, 0).xy;
  let added    = select(noiseVelocity(cell), vec2f(0.0), isAbsorbGhostCell(cell));
  textureStore(velocityOut, cell, vec4f(existing + added, 0.0, 0.0));
}

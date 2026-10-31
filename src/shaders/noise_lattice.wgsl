// Shared by the noise shaders: main.js prepends this file to velocity_noise.wgsl and
// density_inject_noise.wgsl, each of which declares `params: NoiseParams` at binding 2.
//
// Noise is either one random value per cell (blobRadius ≤ 1) or Gaussian blobs of radius
// blobRadius centered on a square lattice of latticeNodesPerSide² nodes, each node with its
// own random value. A cell sums only the nodes within reach, so one pass costs the same
// whatever the radius.

struct NoiseParams {
  gridSize:            u32,
  boundaryMode:        u32,
  seed:                u32,
  latticeNodesPerSide: u32,
  strength:            f32,
  blobRadius:          f32, // in grid cells
  pad0:                f32,
  pad1:                f32,
}

const BOUNDARY_WRAP   = 0u;
const BOUNDARY_ABSORB = 1u;

const PER_CELL_NOISE_MAX_RADIUS = 1.0;
// exp(-9) ≈ 1e-4 of a blob's peak; nodes farther away than this add nothing visible.
const BLOB_REACH_IN_RADII = 3.0;

const U32_RANGE = 4294967296.0;

struct NodeRange {
  first: i32,
  last:  i32,
}

// PCG hash (Jarzynski & Olano 2020): sin-based hashes repeat visibly at large arguments.
fn hashU32(value: u32) -> u32 {
  let state = value * 747796405u + 2891336453u;
  let word  = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

// Uniform in [0, 1), fixed per node and seed.
fn randomUnitAtNode(node: vec2u) -> f32 {
  return f32(hashU32(node.x ^ hashU32(node.y ^ hashU32(params.seed)))) / U32_RANGE;
}

fn usesPerCellNoise() -> bool {
  return params.blobRadius <= PER_CELL_NOISE_MAX_RADIUS;
}

// Absorb boundary cells act as ghost cells, so no injection writes them.
fn isAbsorbGhostCell(cell: vec2i) -> bool {
  let last = i32(params.gridSize) - 1;
  return params.boundaryMode == BOUNDARY_ABSORB
    && (cell.x == 0 || cell.y == 0 || cell.x == last || cell.y == last);
}

fn latticeSpacing() -> f32 {
  return f32(params.gridSize) / f32(params.latticeNodesPerSide);
}

// Node indices along one axis whose blobs reach the given position. In Wrap mode the spacing
// tiles the grid exactly, and at most one period of nodes is taken so each node counts once;
// otherwise nodes 0..latticeNodesPerSide cover both edges.
fn nodeRangeReaching(position: f32) -> NodeRange {
  let spacing = latticeSpacing();
  let reach   = BLOB_REACH_IN_RADII * params.blobRadius;
  let nodes   = i32(params.latticeNodesPerSide);
  let first   = i32(floor((position - reach) / spacing));
  let last    = i32(ceil((position + reach) / spacing));
  if (params.boundaryMode == BOUNDARY_WRAP) {
    return NodeRange(first, min(last, first + nodes - 1));
  }
  return NodeRange(max(first, 0), min(last, nodes));
}

// In Wrap mode the offset is to the node's nearest periodic image.
fn offsetToNode(position: f32, nodeIndex: i32) -> f32 {
  let offset = position - f32(nodeIndex) * latticeSpacing();
  if (params.boundaryMode != BOUNDARY_WRAP) { return offset; }
  let period = f32(params.gridSize);
  return offset - period * round(offset / period);
}

fn nodeId(nodeIndex: i32) -> u32 {
  if (params.boundaryMode != BOUNDARY_WRAP) { return u32(nodeIndex); }
  let nodes = i32(params.latticeNodesPerSide);
  return u32(((nodeIndex % nodes) + nodes) % nodes);
}

fn blobWeight(offset: vec2f) -> f32 {
  return exp(-dot(offset, offset) / (params.blobRadius * params.blobRadius));
}

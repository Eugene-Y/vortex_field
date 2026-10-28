// Jacobi iterations of the 9-point Mehrstellen isotropic pressure solve, several per dispatch.
// Reads pressureIn + divergence, writes the estimate after params.iterations steps to
// pressureOut. The dispatches are chained with ping-pong textures; the total iteration
// count controls incompressibility.
//
// Each workgroup loads a TILE_SIZE² tile of pressure into workgroup memory and iterates
// there. A Jacobi step only reads direct neighbours, so after k steps the outer k rings of
// the tile are stale while the interior is exactly what k separate passes would produce;
// only the interior, (TILE_SIZE − 2·iterations)² cells, is written out. This trades a
// little redundant arithmetic at tile borders for far fewer dispatches and texture reads.

struct PressureParams {
  gridSize:     u32,
  boundaryMode: u32,
  iterations:   u32,   // Jacobi steps in this dispatch; at most MAX_ITERATIONS_PER_DISPATCH
  pad0:         u32,
}

@group(0) @binding(0) var pressureIn:   texture_2d<f32>;
@group(0) @binding(1) var divergenceIn: texture_2d<f32>;
@group(0) @binding(2) var pressureOut:  texture_storage_2d<r32float, write>;
@group(0) @binding(3) var<uniform> params: PressureParams;

const WORKGROUP_WIDTH = 16u;
const CELLS_PER_AXIS  = 2u;   // each invocation owns CELLS_PER_AXIS² tile cells
const TILE_SIZE       = 32u;  // WORKGROUP_WIDTH × CELLS_PER_AXIS
const TILE_CELLS      = 1024u;

const BOUNDARY_WRAP   = 0u;
const BOUNDARY_ABSORB = 1u;

// Two tile buffers: one step reads one and writes the other.
var<workgroup> tileA: array<f32, TILE_CELLS>;
var<workgroup> tileB: array<f32, TILE_CELLS>;

// Integer division is slow on every GPU; callers stay within one period of the grid.
fn wrapIntoGrid(c: i32, n: i32) -> i32 {
  return c + select(select(0, -n, c >= n), n, c < 0);
}

fn insideGrid(cell: vec2i, n: i32) -> bool {
  return cell.x >= 0 && cell.x < n && cell.y >= 0 && cell.y < n;
}

fn outputSize() -> i32 {
  return i32(TILE_SIZE) - 2 * i32(params.iterations);
}

// Grid cell of tile-local cell (0, 0); the tile reaches params.iterations cells beyond
// the block of cells this workgroup writes.
fn tileOrigin(workgroupId: vec2u) -> vec2i {
  return vec2i(workgroupId) * outputSize() - vec2i(i32(params.iterations));
}

fn tileIndex(local: vec2i) -> u32 {
  let clamped = clamp(local, vec2i(0), vec2i(i32(TILE_SIZE) - 1));
  return u32(clamped.y) * TILE_SIZE + u32(clamped.x);
}

// In Wrap mode every tile cell is a real cell of the torus; otherwise cells outside the
// domain are ghosts whose stored value is never read.
fn sourceCell(cell: vec2i, n: i32) -> vec2i {
  if (params.boundaryMode == BOUNDARY_WRAP) { return vec2i(wrapIntoGrid(cell.x, n), wrapIntoGrid(cell.y, n)); }
  return clamp(cell, vec2i(0), vec2i(n - 1));
}

fn readTile(useTileA: bool, index: u32) -> f32 {
  if (useTileA) { return tileA[index]; }
  return tileB[index];
}

fn writeTile(useTileA: bool, index: u32, value: f32) {
  if (useTileA) { tileA[index] = value; } else { tileB[index] = value; }
}

// Neighbour value with the same ghost-cell rules as a single-pass solve:
// Wrap — the tile was loaded periodically; Absorb — Dirichlet p=0 outside the domain;
// Reflect — Neumann, the nearest cell inside the domain.
fn samplePressure(useTileA: bool, local: vec2i, origin: vec2i, n: i32) -> f32 {
  if (params.boundaryMode == BOUNDARY_WRAP) { return readTile(useTileA, tileIndex(local)); }
  let cell = origin + local;
  if (insideGrid(cell, n)) { return readTile(useTileA, tileIndex(local)); }
  if (params.boundaryMode == BOUNDARY_ABSORB) { return 0.0; }
  return readTile(useTileA, tileIndex(clamp(cell, vec2i(0), vec2i(n - 1)) - origin));
}

// Mehrstellen 9-point isotropic Laplacian: cardinal ×4, diagonals ×1.
// Reduces the directional bias that causes diagonal artifacts with the 5-point stencil.
fn jacobiUpdate(useTileA: bool, local: vec2i, origin: vec2i, n: i32, divergence: f32) -> f32 {
  let left   = samplePressure(useTileA, local + vec2i(-1,  0), origin, n);
  let right  = samplePressure(useTileA, local + vec2i( 1,  0), origin, n);
  let bottom = samplePressure(useTileA, local + vec2i( 0, -1), origin, n);
  let top    = samplePressure(useTileA, local + vec2i( 0,  1), origin, n);
  let sw     = samplePressure(useTileA, local + vec2i(-1, -1), origin, n);
  let se     = samplePressure(useTileA, local + vec2i( 1, -1), origin, n);
  let nw     = samplePressure(useTileA, local + vec2i(-1,  1), origin, n);
  let ne     = samplePressure(useTileA, local + vec2i( 1,  1), origin, n);
  return (4.0 * (left + right + bottom + top) + (sw + se + nw + ne) - 6.0 * divergence) / 20.0;
}

fn ownedLocalCell(localId: vec2u, slot: u32) -> vec2i {
  return vec2i(localId + WORKGROUP_WIDTH * vec2u(slot % CELLS_PER_AXIS, slot / CELLS_PER_AXIS));
}

@compute @workgroup_size(16, 16)
fn jacobiPressureBlock(
  @builtin(workgroup_id)        workgroupId: vec3u,
  @builtin(local_invocation_id) localId:     vec3u,
) {
  let n      = i32(params.gridSize);
  let origin = tileOrigin(workgroupId.xy);

  var divergence: array<f32, 4>;
  for (var slot = 0u; slot < CELLS_PER_AXIS * CELLS_PER_AXIS; slot++) {
    let local = ownedLocalCell(localId.xy, slot);
    let cell  = origin + local;
    let source = sourceCell(cell, n);
    tileA[tileIndex(local)] = textureLoad(pressureIn, source, 0).r;
    divergence[slot] = textureLoad(divergenceIn, source, 0).r;
  }
  workgroupBarrier();

  // Cells outside the domain are ghosts in Absorb/Reflect: samplePressure derives them,
  // so only in-domain cells (every cell in Wrap) are iterated.
  var readsTileA = true;
  for (var iteration = 0u; iteration < params.iterations; iteration++) {
    for (var slot = 0u; slot < CELLS_PER_AXIS * CELLS_PER_AXIS; slot++) {
      let local = ownedLocalCell(localId.xy, slot);
      let cell  = origin + local;
      if (params.boundaryMode == BOUNDARY_WRAP || insideGrid(cell, n)) {
        writeTile(!readsTileA, tileIndex(local), jacobiUpdate(readsTileA, local, origin, n, divergence[slot]));
      }
    }
    workgroupBarrier();
    readsTileA = !readsTileA;
  }

  let ring = i32(params.iterations);
  for (var slot = 0u; slot < CELLS_PER_AXIS * CELLS_PER_AXIS; slot++) {
    let local  = ownedLocalCell(localId.xy, slot);
    let cell   = origin + local;
    let inside = all(local >= vec2i(ring)) && all(local < vec2i(ring + outputSize()));
    if (inside && insideGrid(cell, n)) {
      textureStore(pressureOut, cell, vec4f(readTile(readsTileA, tileIndex(local)), 0.0, 0.0, 0.0));
    }
  }
}

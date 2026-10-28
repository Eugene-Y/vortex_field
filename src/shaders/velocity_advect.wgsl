// Semi-Lagrangian advection with Catmull-Rom bicubic interpolation.
// Each thread traces one cell back along the velocity field and samples
// the 4×4 Catmull-Rom neighbourhood at the back-traced position.

struct AdvectParams {
  gridSize:     u32,
  boundaryMode: u32,
  deltaTime:    f32,
  damping:      f32,   // pow(damping, dt) — pre-computed on JS side
}

@group(0) @binding(0) var velocityIn:  texture_2d<f32>;
@group(0) @binding(1) var velocityOut: texture_storage_2d<rg32float, write>;
@group(0) @binding(2) var<uniform> params: AdvectParams;

// Integer division is slow on every GPU; callers stay within one period of the grid.
fn wrapIntoGrid(c: i32, n: i32) -> i32 {
  return c + select(select(0, -n, c >= n), n, c < 0);
}

// Boundary-aware velocity sample at integer cell (col, row).
fn sampleVelocityAt(col: i32, row: i32) -> vec2f {
  let n = i32(params.gridSize);
  if (params.boundaryMode == 0u) {
    return textureLoad(velocityIn, vec2i(wrapIntoGrid(col, n), wrapIntoGrid(row, n)), 0).xy;
  }
  // Absorb: out-of-domain positions contribute zero — no energy smuggled in.
  if (params.boundaryMode == 1u && (col < 0 || col >= n || row < 0 || row >= n)) {
    return vec2f(0.0);
  }
  // Reflect and in-bounds absorb: clamp to nearest interior cell.
  return textureLoad(velocityIn, vec2i(clamp(col, 0, n - 1), clamp(row, 0, n - 1)), 0).xy;
}

// Catmull-Rom weights for fractional offset t in [0, 1].
fn catmullRomWeights(t: f32) -> vec4f {
  let t2 = t * t;
  let t3 = t2 * t;
  return vec4f(
     0.5 * (-t3 + 2.0*t2 - t),
     0.5 * ( 3.0*t3 - 5.0*t2 + 2.0),
     0.5 * (-3.0*t3 + 4.0*t2 + t),
     0.5 * ( t3 - t2),
  );
}

// A back-traced cell can lie several periods away; folding its integer index once keeps
// every stencil index within one period of the grid. The index is an integer-valued f32,
// so the fold is exact, and the fraction is taken from the unfolded position as before.
fn foldCellIndex(cell: f32) -> i32 {
  if (params.boundaryMode != 0u) { return i32(cell); }
  let n = f32(params.gridSize);
  return i32(cell - n * floor(cell / n));
}

// Bicubic Catmull-Rom interpolation at fractional cell position (posX, posY).
// posX and posY are in cell-centre space: posX=0 is the centre of column 0.
fn sampleVelocityBicubic(posX: f32, posY: f32) -> vec2f {
  let iX = foldCellIndex(floor(posX));
  let iY = foldCellIndex(floor(posY));
  let wx = catmullRomWeights(fract(posX));
  let wy = catmullRomWeights(fract(posY));

  var result = vec2f(0.0);
  for (var dr = 0; dr < 4; dr++) {
    var rowSum = vec2f(0.0);
    for (var dc = 0; dc < 4; dc++) {
      rowSum += wx[dc] * sampleVelocityAt(iX + dc - 1, iY + dr - 1);
    }
    result += wy[dr] * rowSum;
  }
  return result;
}

@compute @workgroup_size(8, 8)
fn advectVelocity(@builtin(global_invocation_id) id: vec3u) {
  let col = i32(id.x);
  let row = i32(id.y);
  let n   = i32(params.gridSize);
  if (col >= n || row >= n) { return; }

  let velocity  = textureLoad(velocityIn, vec2i(col, row), 0).xy;
  // Trace back in cell-centre space; velocity units are cells/second.
  let tracedCol = f32(col) - velocity.x * params.deltaTime;
  let tracedRow = f32(row) - velocity.y * params.deltaTime;

  let advected = sampleVelocityBicubic(tracedCol, tracedRow) * params.damping;
  textureStore(velocityOut, vec2i(col, row), vec4f(advected, 0.0, 0.0));
}

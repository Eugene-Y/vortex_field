'use strict';

export class ShaderProgram {
  constructor(gl, vertexSource, fragmentSource) {
    this._gl = gl;
    this._program = compileAndLinkProgram(gl, vertexSource, fragmentSource);
    this._uniformLocations = new Map();
  }

  bind() {
    this._gl.useProgram(this._program);
  }

  setUniform1i(name, value) {
    this._gl.uniform1i(this._resolveUniform(name), value);
  }

  setUniform1f(name, value) {
    this._gl.uniform1f(this._resolveUniform(name), value);
  }

  setUniform2f(name, x, y) {
    this._gl.uniform2f(this._resolveUniform(name), x, y);
  }

  setUniform3f(name, x, y, z) {
    this._gl.uniform3f(this._resolveUniform(name), x, y, z);
  }

  setUniform2fv(name, value) {
    this._gl.uniform2fv(this._resolveUniform(name), value);
  }

  _resolveUniform(name) {
    if (!this._uniformLocations.has(name)) {
      const location = this._gl.getUniformLocation(this._program, name);
      this._uniformLocations.set(name, location);
    }
    return this._uniformLocations.get(name);
  }

  dispose() {
    this._gl.deleteProgram(this._program);
  }
}

function compileAndLinkProgram(gl, vertexSource, fragmentSource) {
  const vertexShader = compileShader(gl, gl.VERTEX_SHADER, vertexSource);
  const fragmentShader = compileShader(gl, gl.FRAGMENT_SHADER, fragmentSource);

  const program = gl.createProgram();
  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);

  gl.deleteShader(vertexShader);
  gl.deleteShader(fragmentShader);

  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const log = gl.getProgramInfoLog(program);
    gl.deleteProgram(program);
    throw new Error(`Shader program link failed:\n${log}`);
  }

  return program;
}

function compileShader(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);

  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    const typeName = type === gl.VERTEX_SHADER ? 'vertex' : 'fragment';
    throw new Error(`${typeName} shader compilation failed:\n${log}`);
  }

  return shader;
}

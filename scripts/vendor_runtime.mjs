// Copies the browser runtime (Transformers.js + the ONNX Runtime WASM backend) into web/vendor/ so the
// page loads nothing from a third-party CDN and works offline.
// Transformers.js bundles its own onnxruntime-web JS; the .wasm must be the exact same build, so the
// version baked into the bundle is checked against the installed onnxruntime-web package.
import { copyFileSync, mkdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';

const OUT = 'web/vendor';
// package.json is not in either package's "exports", so resolve by path instead of require.resolve
const tjsDir = 'node_modules/@huggingface/transformers';
const ortDir = 'node_modules/onnxruntime-web';
const ortVersion = JSON.parse(readFileSync(join(ortDir, 'package.json'), 'utf8')).version;

const bundle = join(tjsDir, 'dist/transformers.min.js');
if (!readFileSync(bundle, 'utf8').includes(ortVersion)) {
  throw new Error(`transformers.min.js was not built against onnxruntime-web ${ortVersion}; reinstall matching versions`);
}

mkdirSync(OUT, { recursive: true });
const files = [
  [bundle, 'transformers.min.js'],
  // asyncify is the build Transformers.js picks for WASM-only devices (see wasmPaths in web/main.js)
  [join(ortDir, 'dist/ort-wasm-simd-threaded.asyncify.mjs'), 'ort-wasm-simd-threaded.asyncify.mjs'],
  [join(ortDir, 'dist/ort-wasm-simd-threaded.asyncify.wasm'), 'ort-wasm-simd-threaded.asyncify.wasm'],
];
for (const [src, name] of files) copyFileSync(src, join(OUT, name));
console.log(`vendored transformers.js + onnxruntime-web ${ortVersion} into ${OUT}/`);

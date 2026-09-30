// Builds dist/ for Cloudflare Workers static assets from web/.
// Workers assets cap each file at 25 MiB, so larger files (the q8 ONNX graphs) are split into
// .partN chunks; worker/index.js reassembles them under their original URL using worker/chunks.json.
import { createHash } from 'node:crypto';
import { cpSync, mkdirSync, readdirSync, readFileSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { join, relative, sep } from 'node:path';

const SRC = 'web';
const DIST = 'dist';
const MANIFEST = 'worker/chunks.json';
const CHUNK_BYTES = 20 * 1024 * 1024;

rmSync(DIST, { recursive: true, force: true });
mkdirSync(DIST, { recursive: true });

const walk = (dir) =>
  readdirSync(dir, { withFileTypes: true }).flatMap((e) => (e.isDirectory() ? walk(join(dir, e.name)) : [join(dir, e.name)]));

const manifest = {};
for (const file of walk(SRC)) {
  const rel = relative(SRC, file);
  const out = join(DIST, rel);
  mkdirSync(join(out, '..'), { recursive: true });
  const size = statSync(file).size;
  if (size <= CHUNK_BYTES) {
    cpSync(file, out);
    continue;
  }
  const data = readFileSync(file);
  const urlPath = '/' + rel.split(sep).join('/');
  const parts = [];
  for (let i = 0, offset = 0; offset < size; i += 1, offset += CHUNK_BYTES) {
    writeFileSync(`${out}.part${i}`, data.subarray(offset, offset + CHUNK_BYTES));
    parts.push(`${urlPath}.part${i}`);
  }
  manifest[urlPath] = { size, parts };
  console.log(`split ${urlPath} (${size} bytes) into ${parts.length} parts`);
}

// Browsers keep ES modules from earlier visits (heuristic caching), so a redeploy could run the old
// main.js against new files. Version the module and stylesheet URLs with a content hash to force a fresh fetch.
const hashOf = (text) => createHash('sha256').update(text).digest('hex').slice(0, 10);
const fileHash = (path) => hashOf(readFileSync(path));
let mainJs = readFileSync(join(SRC, 'main.js'), 'utf8');
for (const dep of ['sentences.js', 'vendor/transformers.min.js']) {
  mainJs = mainJs.replace(`'./${dep}'`, `'./${dep}?v=${fileHash(join(SRC, dep))}'`);
}
writeFileSync(join(DIST, 'main.js'), mainJs);
const indexHtml = readFileSync(join(SRC, 'index.html'), 'utf8')
  .replace('src="main.js"', `src="main.js?v=${hashOf(mainJs)}"`)
  .replace('href="style.css"', `href="style.css?v=${fileHash(join(SRC, 'style.css'))}"`);
writeFileSync(join(DIST, 'index.html'), indexHtml);

mkdirSync('worker', { recursive: true });
writeFileSync(MANIFEST, JSON.stringify(manifest, null, 2) + '\n');
console.log(`wrote ${DIST}/ and ${MANIFEST}`);

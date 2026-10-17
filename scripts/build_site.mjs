// Builds dist/ for Cloudflare Workers static assets from web/.
// Workers assets cap each file at 25 MiB, so larger files (the ONNX graphs) are split into .partN chunks
// under /chunks/; worker/index.js reassembles them under their original URL using worker/chunks.json.
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
  // One /chunks/ prefix lets a single _headers rule cover every part of every model (the file allows
  // 100 rules, and the base + large models together need ~130 parts).
  const chunkDir = join(DIST, 'chunks', rel);
  mkdirSync(join(chunkDir, '..'), { recursive: true });
  const parts = [];
  for (let i = 0, offset = 0; offset < size; i += 1, offset += CHUNK_BYTES) {
    writeFileSync(`${chunkDir}.part${i}`, data.subarray(offset, offset + CHUNK_BYTES));
    parts.push(`/chunks${urlPath}.part${i}`);
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
const shellVersion = hashOf(mainJs + readFileSync(join(SRC, 'style.css'), 'utf8') + readFileSync(join(SRC, 'index.html'), 'utf8') + fileHash(join(SRC, 'vendor/ort-wasm-simd-threaded.asyncify.wasm')));
writeFileSync(join(DIST, 'sw.js'), readFileSync(join(SRC, 'sw.js'), 'utf8').replace('{{SHELL_VERSION}}', shellVersion));

// Without cross-origin isolation ONNX Runtime refuses SharedArrayBuffer and runs on one WASM thread
// (measured: ~1.8× slower per sentence, ~2.3× under 4× CPU throttling). Everything the page loads is
// same-origin (runtime vendored, model reassembled by the Worker), so require-corp blocks nothing.
// Workers assets default to max-age=0, so every chunk revalidated at the origin on each first load
// (cf-cache-status MISS/REVALIDATED). The .partN files only change together with the chunk list and
// the bytes behind them, so they can be cached for a year; the assembled URLs keep revalidating.
writeFileSync(
  join(DIST, '_headers'),
  '/*\n  Cross-Origin-Opener-Policy: same-origin\n  Cross-Origin-Embedder-Policy: require-corp\n'
    + '/chunks/*\n  Cache-Control: public, max-age=31536000, immutable\n',
);

// Training statistics on the page come from out/rounds.json (one entry per round, written after its
// evaluation), so the copy cannot drift from the published model.
const rounds = JSON.parse(readFileSync('out/rounds.json', 'utf8'));
// The live round is marked explicitly: after a regressing round the previous model can be put back live.
const live = rounds.find((r) => r.live);
if (!live?.data || !live.remaining) throw new Error('out/rounds.json needs exactly one round with "live": true, "data" and "remaining"');
const de = (n) => n.toLocaleString('de-DE');
const pct = (x) => `${(x * 100).toLocaleString('de-DE', { minimumFractionDigits: 1, maximumFractionDigits: 1 })}&nbsp;%`;
const [accErr, accTotal] = live.accounts.split('/').map(Number);
// Both live models are highlighted: the default (base) and the optional large one.
const isLive = (r) => r === live || r.variant === 'large';
const row = (r) => `<tr${isLive(r) ? ' class="live"' : ''}>`
  + `<td>${r.round}</td><td>${r.model}</td><td>${r.change}</td>`
  + `<td class="num">${pct(r.test)}</td>`
  + `<td class="num">${r.accounts ? r.accounts.replace('/', '&nbsp;von&nbsp;') : '–'}</td>`
  + `<td>${r === live ? 'live (Standard)' : r.variant === 'large' ? 'live (genauer)' : r.deployed ? 'war live' : r.release ? 'nur Release' : '–'}</td></tr>`;
const dirBytes = (dir) => walk(join(SRC, dir)).reduce((n, f) => n + statSync(f).size, 0);
// Under 1 GB the page says "rund 680 MB"; above, "rund 1,9 GB" reads better than "1.950 MB".
const dirSize = (dir) => {
  const bytes = dirBytes(dir);
  return bytes < 1e9
    ? `${de(Math.round(bytes / 1e6 / 10) * 10)}&nbsp;MB`
    : `${(bytes / 1e9).toLocaleString('de-DE', { maximumFractionDigits: 1 })}&nbsp;GB`;
};
// The large model is optional on the page: out/rounds.json marks its round with "variant": "large".
const large = rounds.find((r) => r.variant === 'large');
if (!large) throw new Error('out/rounds.json needs one round with "variant": "large" for web/model-large');
const stats = {
  MODEL_SIZE: dirSize('model'),
  MODEL_LARGE_SIZE: dirSize('model-large'),
  LARGE_TEST: pct(large.test),
  LARGE_ACC_ERR: large.accounts.split('/')[0],
  ROUND: String(live.round),
  ROUNDS: String(rounds.length),
  TATOEBA: de(Math.round(live.data.tatoeba / 1000) * 1000),
  GENERATED: de(Math.round(live.data.agy / 1000) * 1000),
  NARRATIVES: String(live.data.narratives),
  TRAIN: de(live.train),
  TEST: pct(live.test),
  ACC_ERR: String(accErr),
  ACC_TOTAL: String(accTotal),
  ACC_PCT: pct(accErr / accTotal),
  REMAINING: live.remaining,
  ROUNDS_TABLE: rounds.map(row).join('\n'),
};
let indexHtml = readFileSync(join(SRC, 'index.html'), 'utf8')
  .replace('src="main.js"', `src="main.js?v=${hashOf(mainJs)}"`)
  .replace('href="style.css"', `href="style.css?v=${fileHash(join(SRC, 'style.css'))}"`);
indexHtml = indexHtml.replace(/\{\{(\w+)\}\}/g, (m, key) => {
  if (!(key in stats)) throw new Error(`index.html uses unknown placeholder ${m}`);
  return stats[key];
});
writeFileSync(join(DIST, 'index.html'), indexHtml);

mkdirSync('worker', { recursive: true });
writeFileSync(MANIFEST, JSON.stringify(manifest, null, 2) + '\n');
console.log(`wrote ${DIST}/ and ${MANIFEST}`);

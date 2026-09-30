import { readFileSync } from 'node:fs';
import { pipeline, env } from '@huggingface/transformers';

env.localModelPath = 'web/';
env.allowRemoteModels = false;

const loadJsonl = (path) =>
  readFileSync(path, 'utf8')
    .split('\n')
    .filter((line) => line.trim())
    .map((line) => JSON.parse(line));

const challenge = loadJsonl('data/challenge_20.jsonl');
const reference = loadJsonl('out/pred_challenge.jsonl');

// device:'wasm' is browser-only; Node runs the same ONNX graphs and dtypes on onnxruntime-node (cpu).
// Same dtype map as web/main.js; progress_callback makes Node take the same size-prefetch path as the browser.
const pipe = await pipeline('text2text-generation', 'model', {
  dtype: { model: 'fp32', encoder_model: 'fp32', decoder_model_merged: 'q8' },
  progress_callback: () => {},
});

const preds = [];
for (const row of challenge) {
  const [{ generated_text }] = await pipe(row.source, { max_new_tokens: 64, do_sample: false });
  preds.push(generated_text);
}

console.log(preds[0]);

if (reference.length !== challenge.length) {
  console.error(`out/pred_challenge.jsonl has ${reference.length} lines, expected ${challenge.length}`);
  process.exit(1);
}

let identical = 0;
const mismatches = [];
for (let i = 0; i < challenge.length; i += 1) {
  const ref = reference[i];
  if (ref.source === challenge[i].source && ref.pred === preds[i]) {
    identical += 1;
  } else {
    mismatches.push({ line: i + 1, source: challenge[i].source, node: preds[i], pytorch: ref.pred });
  }
}

console.log(`identical to PyTorch: ${identical}/${challenge.length}`);
for (const m of mismatches) {
  console.log(
    `line ${m.line} ${JSON.stringify(m.source)}\n  node:    ${JSON.stringify(m.node)}\n  pytorch: ${JSON.stringify(m.pytorch)}`,
  );
}

process.exit(identical >= 18 ? 0 : 1);

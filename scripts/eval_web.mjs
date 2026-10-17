// Test exact match of the shipped web model (Transformers.js, same dtypes as web/main.js) on data/test.jsonl.
// PyTorch numbers come from the fp32 checkpoint; this is what the page actually runs (q8 decoder).
//   node scripts/eval_web.mjs [model-dir under web/, default "model"]
import { readFileSync } from 'node:fs';
import { pipeline, env } from '@huggingface/transformers';

env.localModelPath = 'web/';
env.allowRemoteModels = false;

const model = process.argv[2] ?? 'model';
const normalize = (t) => {
  t = t.trim();
  return (t.endsWith('.') ? t.slice(0, -1) : t).split(/\s+/).join(' ');
};
const rows = readFileSync('data/test.jsonl', 'utf8').split('\n').filter((l) => l.trim()).map((l) => JSON.parse(l));
const pipe = await pipeline('text2text-generation', model, {
  dtype: { model: 'fp32', encoder_model: 'fp32', decoder_model_merged: 'q8' },
  progress_callback: () => {},
});
let hits = 0;
const t0 = Date.now();
for (const row of rows) {
  const [{ generated_text }] = await pipe(row.source, { max_new_tokens: 64, do_sample: false });
  if (normalize(generated_text) === normalize(row.target)) hits += 1;
}
console.log(`${model}: ${hits}/${rows.length} exact (${((Date.now() - t0) / rows.length).toFixed(0)} ms/sentence)`);

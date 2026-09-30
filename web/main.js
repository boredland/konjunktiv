// Runtime is served from this origin (copied in by scripts/build_site.mjs), so no third-party CDN is
// contacted and the page keeps working offline once the browser has cached it.
import { pipeline, env, ModelRegistry } from './vendor/transformers.min.js';
import { splitSentences } from './sentences.js';

const TASK = 'text2text-generation';
const MODEL = 'model';

// The browser build defaults allowLocalModels to false; enable it so 'model' resolves to ./model/
env.allowLocalModels = true;
env.localModelPath = './';
env.allowRemoteModels = false;
env.backends.onnx.wasm.wasmPaths = {
  mjs: new URL('./vendor/ort-wasm-simd-threaded.asyncify.mjs', import.meta.url).href,
  wasm: new URL('./vendor/ort-wasm-simd-threaded.asyncify.wasm', import.meta.url).href,
};
// With a progress callback, Transformers.js GETs each model file once just to read Content-Length
// and never reads that body, then GETs it again for real. Chrome's HTTP cache lets the second request
// wait on the first for the same URL, which stalled loading of the 100+ MB files. The library keeps
// its own Cache API copy ("transformers-cache"), so skipping the HTTP cache costs no re-downloads.
env.fetch = (url, init) => fetch(url, { ...init, cache: 'no-store' });

const PIPELINE_OPTIONS = {
  // fp32 encoder + q8 decoder: see scripts/quantize.py for the measured tradeoff.
  // Transformers.js resolves the encoder dtype by session name ("model") when prefetching sizes
  // for progress, but by file name ("encoder_model") when loading; with only one key the other
  // path falls back to q8 and requests a missing encoder_model_quantized.onnx.
  dtype: { model: 'fp32', encoder_model: 'fp32', decoder_model_merged: 'q8' },
  device: 'wasm',
};

const $ = (id) => document.getElementById(id);
const loadButton = $('load');
const forgetButton = $('forget');
const progressBar = $('progress');
const loadStatus = $('load-status');
const runButton = $('run');
const runStatus = $('run-status');
const runProgress = $('run-progress');
const direct = $('direct');
const indirect = $('indirect');

let pipe = null;

function setStatus(el, message, kind = '') {
  el.textContent = message;
  el.className = `status ${kind}`.trim();
}

function fitToContent(el) {
  el.style.height = 'auto';
  el.style.height = `${el.scrollHeight}px`;
}

const formatMB = (bytes) => `${Math.round(bytes / 1e6)} MB`;
const nextPaint = () => new Promise((resolve) => requestAnimationFrame(() => setTimeout(resolve, 0)));

// Disables the button and swaps its label while work runs; the label element keeps the spinner span intact.
function setBusy(button, busy, label) {
  button.disabled = busy;
  button.classList.toggle('is-busy', busy);
  button.setAttribute('aria-busy', String(busy));
  if (label) button.querySelector('.label').textContent = label;
}

async function loadModel() {
  setBusy(loadButton, true, 'Wird geladen …');
  runButton.disabled = true;
  forgetButton.hidden = true;
  progressBar.hidden = false;
  progressBar.value = 0;
  // Model and tokenizer files load in parallel: when one fails, the others keep reporting
  // progress after the catch has written the error and would overwrite it.
  let loading = true;
  try {
    pipe = await pipeline(TASK, MODEL, {
      ...PIPELINE_OPTIONS,
      progress_callback: (p) => {
        if (!loading || p.status !== 'progress_total') return;
        progressBar.value = p.progress;
        setStatus(loadStatus, `Lade Modell … ${Math.round(p.progress)} % (${formatMB(p.loaded)} von ${formatMB(p.total)})`);
      },
    });
    loading = false;
    progressBar.value = 100;
    progressBar.hidden = true;
    setStatus(loadStatus, 'Modell ist geladen und auf diesem Gerät gespeichert. Die Umwandlung läuft jetzt offline.', 'ok');
    setBusy(loadButton, false, 'Modell geladen');
    loadButton.disabled = true;
    forgetButton.hidden = false;
    runButton.disabled = false;
    setStatus(runStatus, 'Bereit.');
  } catch (error) {
    loading = false;
    progressBar.hidden = true;
    setBusy(loadButton, false, 'Modell herunterladen');
    setStatus(loadStatus, `Fehler beim Laden: ${error.message}`, 'err');
  }
}

async function forgetModel() {
  forgetButton.disabled = true;
  try {
    await pipe?.dispose();
    pipe = null;
    const { filesDeleted } = await ModelRegistry.clear_pipeline_cache(TASK, MODEL, PIPELINE_OPTIONS);
    runButton.disabled = true;
    setBusy(loadButton, false, 'Modell herunterladen');
    forgetButton.hidden = true;
    setStatus(loadStatus, `Gespeichertes Modell gelöscht (${filesDeleted} Dateien).`);
    setStatus(runStatus, 'Laden Sie zuerst das Modell (Schritt 1).');
  } catch (error) {
    setStatus(loadStatus, `Fehler beim Löschen: ${error.message}`, 'err');
  } finally {
    forgetButton.disabled = false;
  }
}

function stripQuotes(text) {
  for (const [open, close] of [['„', '“'], ['"', '"']]) {
    if (text.startsWith(open) && text.endsWith(close) && text.length >= open.length + close.length) {
      return text.slice(open.length, text.length - close.length).trim();
    }
  }
  return text;
}

async function convert() {
  const speaker = $('speaker').value;
  const text = stripQuotes(direct.value.trim());
  if (!text) {
    setStatus(runStatus, 'Bitte direkte Rede eingeben.', 'err');
    return;
  }
  setBusy(runButton, true, 'Wird umgewandelt …');
  indirect.value = '';
  fitToContent(indirect);
  const sentences = splitSentences(text);
  runProgress.max = sentences.length;
  runProgress.value = 0;
  runProgress.hidden = sentences.length < 2;
  try {
    const converted = [];
    for (const [i, sentence] of sentences.entries()) {
      setStatus(runStatus, `Wandle um: Satz ${i + 1} von ${sentences.length} …`);
      // WASM inference blocks the main thread; without a painted frame first, the spinner, progress
      // bar and status never appear because the browser only repaints after the whole run.
      await nextPaint();
      const [{ generated_text }] = await pipe(`sprecher: ${speaker} | ${sentence}`, {
        max_new_tokens: 64,
        do_sample: false,
      });
      converted.push(generated_text);
      indirect.value = converted.join(' ');
      fitToContent(indirect);
      runProgress.value = i + 1;
    }
    setStatus(runStatus, 'Fertig.', 'ok');
  } catch (error) {
    setStatus(runStatus, `Fehler: ${error.message}`, 'err');
  } finally {
    setBusy(runButton, false, 'In indirekte Rede umwandeln');
    runProgress.hidden = true;
  }
}

async function init() {
  loadButton.addEventListener('click', loadModel);
  forgetButton.addEventListener('click', forgetModel);
  runButton.addEventListener('click', convert);
  // Without it, the page itself (not the model) needs a network connection on the next visit.
  navigator.serviceWorker?.register('./sw.js').catch(() => {});
  // Reopening the page after a download should not ask for another click: load from the local cache.
  try {
    if (await ModelRegistry.is_pipeline_cached(TASK, MODEL, PIPELINE_OPTIONS)) {
      setStatus(loadStatus, 'Modell ist bereits auf diesem Gerät gespeichert – wird aus dem Speicher geladen …');
      await loadModel();
    }
  } catch {
    // no Cache API (e.g. private mode): fall back to the explicit download button
  }
}

init();

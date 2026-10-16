// Runtime is served from this origin (copied in by scripts/build_site.mjs), so no third-party CDN is
// contacted and the page keeps working offline once the browser has cached it.
import { pipeline, env, ModelRegistry } from './vendor/transformers.min.js';
import { splitParagraphs, splitSentences } from './sentences.js';

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
// With a progress callback, Transformers.js first GETs each model file with "Range: bytes=0-0" just
// to read its size. Neither Workers assets nor the chunk Worker honour Range, so that probe streamed
// ~58 MB of each ONNX file before being cancelled (measured 794 MB received for a 680 MB model).
// A HEAD returns the same Content-Length with no body. The real download skips the HTTP cache: Chrome
// otherwise queues it behind the probe for the same URL, and the library keeps its own Cache API copy.
env.fetch = (url, init = {}) => {
  const range = new Headers(init.headers).get('range');
  return fetch(url, { ...init, cache: 'no-store', ...(range === 'bytes=0-0' ? { method: 'HEAD' } : {}) });
};

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
const copyButton = $('copy');
const copyStatus = $('copy-status');
const exampleRow = $('example-row');
const exampleButton = $('example');
const exampleStatus = $('example-status');
const exampleProgress = $('example-progress');

let pipe = null;

// Chrome's built-in Gemini Nano (Prompt API) writes example input on the device, so it keeps the page's
// promise that no text leaves the browser. Chrome downloads the model only after a site asks for it, so
// the button is also offered while it is still "downloadable"; its label then names the download.
const EXAMPLE_OPTIONS = {
  expectedInputs: [{ type: 'text', languages: ['de'] }],
  expectedOutputs: [{ type: 'text', languages: ['de'] }],
};
// Asylum-hearing accounts are the case the model was audited on (see the training section on the page).
// Short sentences because the converter caps each sentence at 64 tokens; the rest mirrors the limits
// stated on the page: statements only, no questions, commands or du/ihr address.
const EXAMPLE_SYSTEM_PROMPT = [
  'Du schreibst Beispieltexte für ein Werkzeug, das direkte Rede in indirekte Rede umwandelt.',
  'Schreibe einen erfundenen Erfahrungsbericht von etwa 100 Wörtern, wie ihn eine asylsuchende Person',
  'in ihrer Anhörung beim Bundesamt für Migration und Flüchtlinge in der Ich-Form erzählt:',
  'was ihr in der Heimat passiert ist, warum sie geflohen ist und wie sie nach Deutschland kam.',
  'Sachlich und konkret, mit Orten, Zeitangaben und Personen aus ihrem Umfeld. Mische Gegenwart und Vergangenheit.',
  'Kurze Sätze mit höchstens 20 Wörtern. Nur Aussagesätze: keine Fragen, keine Befehle,',
  'niemand wird mit du oder ihr angesprochen.',
  'Antworte nur mit dem Bericht als einem Absatz, ohne Anführungszeichen, Überschrift oder Erklärung.',
].join(' ');
// The web Prompt API exposes no temperature, so variety comes from a different flight reason per request.
const EXAMPLE_TOPICS = [
  'politische Verfolgung nach einer Demonstration', 'religiöse Verfolgung', 'drohende Zwangsrekrutierung',
  'Bedrohung durch eine bewaffnete Gruppe', 'Verfolgung wegen der sexuellen Orientierung',
  'Verfolgung als Journalistin oder Journalist', 'Blutrache zwischen zwei Familien', 'Bürgerkrieg in der Heimatstadt',
  'drohende Zwangsheirat', 'Verfolgung wegen der Zugehörigkeit zu einer ethnischen Minderheit',
];
const EXAMPLE_LABEL = 'Beispiel erzeugen';
let exampleIdleLabel = EXAMPLE_LABEL;
let exampleSession = null;

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
    // Private windows cap Cache Storage: the 439 MB encoder then fails to store while the model still runs
    // from memory, so only claim "stored" when the files really are in the cache.
    const stored = await ModelRegistry.is_pipeline_cached(TASK, MODEL, PIPELINE_OPTIONS).catch(() => false);
    setStatus(
      loadStatus,
      stored
        ? 'Modell ist geladen und auf diesem Gerät gespeichert. Die Umwandlung läuft jetzt offline.'
        : 'Modell ist geladen, die Umwandlung läuft offline. Speichern war nicht möglich (etwa im privaten Fenster) – beim nächsten Besuch wird es neu heruntergeladen.',
      'ok',
    );
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
  copyButton.disabled = true;
  setStatus(copyStatus, '');
  indirect.value = '';
  fitToContent(indirect);
  const paragraphs = splitParagraphs(text).map((p) => ({ ...p, sentences: splitSentences(p.text), out: [] }));
  const total = paragraphs.reduce((n, p) => n + p.sentences.length, 0);
  // Rebuilds the output with the input's own line breaks, so paragraphs and lists stay intact.
  const render = () => paragraphs.map((p) => p.out.join(' ') + (p.out.length ? p.sep : '')).join('').trimEnd();
  runProgress.max = total;
  runProgress.value = 0;
  runProgress.hidden = total < 2;
  try {
    let done = 0;
    for (const paragraph of paragraphs) {
      for (const sentence of paragraph.sentences) {
        setStatus(runStatus, `Wandle um: Satz ${done + 1} von ${total} …`);
        // WASM inference blocks the main thread; without a painted frame first, the spinner, progress
        // bar and status never appear because the browser only repaints after the whole run.
        await nextPaint();
        const [{ generated_text }] = await pipe(`sprecher: ${speaker} | ${sentence}`, {
          max_new_tokens: 64,
          do_sample: false,
        });
        paragraph.out.push(generated_text);
        indirect.value = render();
        fitToContent(indirect);
        runProgress.value = ++done;
      }
    }
    setStatus(runStatus, 'Fertig.', 'ok');
  } catch (error) {
    setStatus(runStatus, `Fehler: ${error.message}`, 'err');
  } finally {
    setBusy(runButton, false, 'In indirekte Rede umwandeln');
    runProgress.hidden = true;
    copyButton.disabled = !indirect.value;
  }
}

async function copyOutput() {
  try {
    await navigator.clipboard.writeText(indirect.value);
  } catch {
    // The async clipboard needs a secure context and permission; fall back to selecting the text so the
    // user can copy it with the keyboard instead of getting nothing.
    indirect.select();
    setStatus(copyStatus, 'Kopieren nicht möglich – Text ist markiert, bitte Strg+C drücken.', 'err');
    return;
  }
  setStatus(copyStatus, 'Kopiert.', 'ok');
  setTimeout(() => setStatus(copyStatus, ''), 2000);
}

async function offerExamples() {
  if (!('LanguageModel' in self)) return;
  let availability;
  try {
    availability = await LanguageModel.availability(EXAMPLE_OPTIONS);
  } catch {
    return;
  }
  if (availability === 'unavailable') return;
  if (availability !== 'available') {
    exampleIdleLabel = `${EXAMPLE_LABEL} (lädt Gemini Nano)`;
    exampleButton.querySelector('.label').textContent = exampleIdleLabel;
    setStatus(exampleStatus, 'Chrome lädt dafür beim ersten Klick einmalig sein Sprachmodell herunter, mehrere GB.');
  }
  exampleRow.hidden = false;
}

async function writeExample() {
  setBusy(exampleButton, true, exampleSession ? 'Wird geschrieben …' : 'Wird vorbereitet …');
  setStatus(exampleStatus, '');
  try {
    // create() is the first await: a "downloadable" model only downloads while the click's user activation lasts.
    exampleSession ??= await LanguageModel.create({
      ...EXAMPLE_OPTIONS,
      initialPrompts: [{ role: 'system', content: EXAMPLE_SYSTEM_PROMPT }],
      monitor(m) {
        m.addEventListener('downloadprogress', (e) => {
          exampleProgress.hidden = false;
          exampleProgress.value = e.loaded;
          // loaded reaches 1 before the model is unpacked and loaded, which takes a while longer.
          setStatus(exampleStatus, e.loaded < 1
            ? `Lade Gemini Nano … ${Math.floor(e.loaded * 100)} %`
            : 'Gemini Nano ist geladen und wird vorbereitet …');
        });
      },
    });
    exampleIdleLabel = EXAMPLE_LABEL;
    exampleProgress.hidden = true;
    setBusy(exampleButton, true, 'Wird geschrieben …');
    setStatus(exampleStatus, '');
    direct.value = '';
    // One clone per example: in a single long session the earlier examples steer the next ones.
    const session = await exampleSession.clone();
    try {
      const topic = EXAMPLE_TOPICS[Math.floor(Math.random() * EXAMPLE_TOPICS.length)];
      for await (const chunk of session.promptStreaming(`Thema: ${topic}`)) {
        direct.value += chunk;
        fitToContent(direct);
      }
    } finally {
      session.destroy();
    }
    direct.value = stripQuotes(direct.value.trim());
    fitToContent(direct);
    setStatus(exampleStatus, 'Geschrieben von Gemini Nano, lokal in Chrome.', 'ok');
  } catch (error) {
    exampleProgress.hidden = true;
    setStatus(exampleStatus, `Fehler: ${error.message}`, 'err');
  } finally {
    setBusy(exampleButton, false, exampleIdleLabel);
  }
}

async function init() {
  loadButton.addEventListener('click', loadModel);
  forgetButton.addEventListener('click', forgetModel);
  runButton.addEventListener('click', convert);
  copyButton.addEventListener('click', copyOutput);
  exampleButton.addEventListener('click', writeExample);
  offerExamples();
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

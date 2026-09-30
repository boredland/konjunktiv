// The model was trained on single sentences capped at 64 tokens; a paragraph in one call gets
// cut off mid-word, so input is converted sentence by sentence.

// A period after these does not end a sentence ("z. B. Tom", "Dr. Müller").
const ABBREVIATIONS = new Set([
  'bzw', 'ca', 'd', 'dr', 'etc', 'evtl', 'ggf', 'hr', 'fr', 'inkl', 'max', 'min', 'nr', 'prof', 'str',
  'u', 'usw', 'vgl', 'z', 'b', 'a', 'o', 'ä', 'bspw', 'sog', 'St', 'st',
]);

export function splitSentences(text) {
  const sentences = [];
  let start = 0;
  // candidate boundary: . ! or ? (plus closing quotes), whitespace, then an uppercase letter or opening quote
  const boundary = /[.!?]["“”»«']?\s+(?=["„“»«']?[A-ZÄÖÜ])/g;
  for (const match of text.matchAll(boundary)) {
    const before = text.slice(start, match.index);
    const lastWord = before.split(/\s+/).pop() ?? '';
    // "26. September", "1. Mai": a period after a number is an ordinal, not a sentence end
    if (text[match.index] === '.' && (/\d$/.test(lastWord) || ABBREVIATIONS.has(lastWord.toLowerCase()))) {
      continue;
    }
    const end = match.index + match[0].trimEnd().length;
    sentences.push(text.slice(start, end).trim());
    start = match.index + match[0].length;
  }
  const rest = text.slice(start).trim();
  if (rest) sentences.push(rest);
  return sentences.filter(Boolean);
}

// Line breaks carry the text's structure (paragraphs, lists), so they are kept verbatim: returns
// [{ text, sep }] where `sep` is the exact whitespace that followed the block in the input.
export function splitParagraphs(input) {
  const parts = input.split(/(\s*\n\s*)/);
  const blocks = [];
  for (let i = 0; i < parts.length; i += 2) {
    const text = parts[i].trim();
    const sep = parts[i + 1] ?? '';
    if (text) {
      blocks.push({ text, sep: sep.replace(/[^\n]/g, '') });
    } else if (blocks.length && sep) {
      // blank line between blocks: add its newlines to the previous separator
      blocks[blocks.length - 1].sep += sep.replace(/[^\n]/g, '');
    }
  }
  if (blocks.length) blocks[blocks.length - 1].sep = '';
  return blocks;
}

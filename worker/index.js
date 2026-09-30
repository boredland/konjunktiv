// Only requests that match no static asset reach this Worker; the chunked model files are the
// only such paths (see scripts/build_site.mjs), everything else is served from dist/ directly.
import chunks from './chunks.json';

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    const entry = chunks[url.pathname];
    if (!entry || (request.method !== 'GET' && request.method !== 'HEAD')) {
      return env.ASSETS.fetch(request);
    }

    // WebAssembly.instantiateStreaming rejects anything but application/wasm
    const contentType = url.pathname.endsWith('.wasm') ? 'application/wasm' : 'application/octet-stream';
    const headers = { 'Content-Type': contentType, 'Content-Length': String(entry.size) };
    if (request.method === 'HEAD') {
      return new Response(null, { headers });
    }

    // FixedLengthStream sets Content-Length, which Transformers.js uses for load progress.
    const { readable, writable } = new FixedLengthStream(entry.size);
    ctx.waitUntil(
      (async () => {
        try {
          for (const part of entry.parts) {
            const res = await env.ASSETS.fetch(new URL(part, url));
            if (!res.ok) throw new Error(`${part}: HTTP ${res.status}`);
            await res.body.pipeTo(writable, { preventClose: true });
          }
          await writable.close();
        } catch (error) {
          await writable.abort(error);
        }
      })(),
    );
    return new Response(readable, { headers: { 'Content-Type': contentType } });
  },
};

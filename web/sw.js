// Keeps the page itself (HTML, JS, CSS, WASM runtime) available offline. Model files are not handled
// here: Transformers.js stores them in its own Cache Storage bucket ("transformers-cache").
const CACHE = 'konjunktiv-shell-v1';

self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', (event) => event.waitUntil(self.clients.claim()));

self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);
  if (event.request.method !== 'GET' || url.origin !== self.location.origin || url.pathname.startsWith('/model/')) {
    return;
  }
  // Network first so a redeploy shows up immediately; the cached copy is the offline fallback.
  event.respondWith(
    fetch(event.request)
      .then((response) => {
        if (response.ok) {
          const copy = response.clone();
          event.waitUntil(caches.open(CACHE).then((cache) => cache.put(event.request, copy)));
        }
        return response;
      })
      .catch(async () => (await caches.match(event.request, { ignoreSearch: false })) ?? Response.error()),
  );
});

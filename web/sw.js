// Keeps the page itself (HTML, JS, CSS, WASM runtime) available offline. Model files are not handled
// here: Transformers.js stores them in its own Cache Storage bucket ("transformers-cache").
// scripts/build_site.mjs replaces the version with a hash of the shell files, so every redeploy opens a
// fresh cache; older ones (with their superseded ?v= module URLs) are deleted on activate.
const CACHE = 'konjunktiv-shell-{{SHELL_VERSION}}';

self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', (event) =>
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k.startsWith('konjunktiv-shell-') && k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim()),
  ),
);

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

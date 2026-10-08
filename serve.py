#!/usr/bin/env python3
"""Static dev server that sets the cross-origin-isolation headers required for wasm threads.

`SharedArrayBuffer` (and therefore wasm-bindgen-rayon multithreading) only works on a
cross-origin-isolated page, which needs these two response headers on every request:

    Cross-Origin-Opener-Policy: same-origin
    Cross-Origin-Embedder-Policy: require-corp   (or credentialless, which this server sends)

Python's plain `http.server` doesn't send them, so use this instead:

    python3 serve.py [port] [host]   # default 8080 on 127.0.0.1 (also: ./run.sh)

Pass host 0.0.0.0 to make the server reachable from other machines on the network.

For static production hosts that can't set headers (e.g. GitHub Pages), use the
`coi-serviceworker` shim in index.html instead — this server is for local dev.
"""
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


class COIHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        # `credentialless` (not `require-corp`) still enables SharedArrayBuffer but lets
        # cross-origin subresources (tagify/unpkg, Google fonts) load without CORP headers —
        # matching the coi-serviceworker config used on the deployed (GitHub Pages) site.
        self.send_header("Cross-Origin-Embedder-Policy", "credentialless")
        # Dev convenience: never cache, so rebuilt wasm/JS is always picked up.
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    # Loopback by default, so the dev server is not exposed to the local network.
    host = sys.argv[2] if len(sys.argv) > 2 else "127.0.0.1"
    print(f"Serving cross-origin-isolated on http://{host}:{port}  (COOP/COEP + no-cache)")
    print("Press Ctrl+C to stop")
    ThreadingHTTPServer((host, port), COIHandler).serve_forever()

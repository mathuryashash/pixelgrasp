"""Local static server for the demo (correct MIME types for .mjs/.wasm/.onnx, no caching).
    python web/tools/serve.py [port]      -> http://localhost:8765/
"""
import http.server, socketserver, sys, os, functools

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


class H(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map,
                      ".js": "text/javascript", ".mjs": "text/javascript", ".wasm": "application/wasm",
                      ".onnx": "application/octet-stream", ".json": "application/json", ".xml": "text/xml",
                      ".stl": "application/octet-stream", ".svg": "image/svg+xml"}

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, *a):
        pass


socketserver.ThreadingTCPServer.allow_reuse_address = True
with socketserver.ThreadingTCPServer(("127.0.0.1", PORT), functools.partial(H, directory=ROOT)) as s:
    print(f"serving {os.path.abspath(ROOT)} on http://localhost:{PORT}/", flush=True)
    s.serve_forever()

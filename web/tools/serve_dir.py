"""Serve an assembled static site dir (like GitHub Pages would). python serve_dir.py <dir> <port>"""
import http.server, socketserver, sys, functools
D, PORT = sys.argv[1], int(sys.argv[2])


class H(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map, ".js": "text/javascript", ".mjs": "text/javascript",
                      ".wasm": "application/wasm", ".onnx": "application/octet-stream", ".json": "application/json", ".xml": "text/xml"}

    def log_message(self, *a):
        pass


socketserver.ThreadingTCPServer.allow_reuse_address = True
with socketserver.ThreadingTCPServer(("127.0.0.1", PORT), functools.partial(H, directory=D)) as s:
    print(f"serving {D} on http://localhost:{PORT}/", flush=True)
    s.serve_forever()

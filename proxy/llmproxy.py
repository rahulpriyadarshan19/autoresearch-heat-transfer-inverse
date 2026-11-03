"""Local forwarder to the Anthropic Messages API; it alone reads the API key. Runs as user llmkey (Python 3.8)."""

import http.server
import json
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

CONFIG_FILE = "/etc/llmproxy/config.json"
KEY_FILE = "/etc/llmproxy/key"
UPSTREAM = "https://api.anthropic.com/v1/messages"  # fixed: a request can never choose where the key goes
MAX_BODY = 1 << 20
REQUEST_HEADERS = ("content-type", "anthropic-version", "anthropic-beta")
RESPONSE_HEADERS = ("content-type", "request-id", "retry-after")

with open(CONFIG_FILE) as fh:
    CONFIG = json.load(fh)
with open(KEY_FILE) as fh:
    KEY = fh.read().strip()
TLS = ssl.create_default_context()


class Handler(http.server.BaseHTTPRequestHandler):
    server_version = "llmproxy"
    sys_version = ""

    def log_message(self, fmt, *args):
        pass

    def _note(self, status, model, t0):
        sys.stderr.write("{} {} {:.2f}s\n".format(status, model, time.time() - t0))

    def _reply(self, status, body, headers=()):
        self.send_response(status)
        for k, v in headers:
            self.send_header(k, v)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status, message, t0, model="-"):
        body = json.dumps({"type": "error", "error": {"type": "proxy_error", "message": message}}).encode()
        self._reply(status, body, [("content-type", "application/json")])
        self._note(status, model, t0)

    def do_POST(self):
        t0 = time.time()
        url = urllib.parse.urlsplit(self.path)
        if url.path != "/v1/messages" or url.query not in ("", "beta=true"):
            return self._error(404, "only POST /v1/messages is forwarded", t0)
        try:
            n = int(self.headers.get("content-length", "0"))
        except ValueError:
            return self._error(400, "bad content-length", t0)
        if n <= 0 or n > MAX_BODY:
            return self._error(413, "body size not allowed", t0)
        body = self.rfile.read(n)
        try:
            model = json.loads(body.decode()).get("model", "")
        except ValueError:
            return self._error(400, "body is not JSON", t0)
        if model not in CONFIG["models"]:
            return self._error(403, "model not in the proxy allowlist", t0, model)
        betas = [b.strip() for b in self.headers.get("anthropic-beta", "").split(",") if b.strip()]
        if any(b not in CONFIG["betas"] for b in betas):
            return self._error(403, "beta header not in the proxy allowlist", t0, model)
        headers = {k: self.headers[k] for k in REQUEST_HEADERS if self.headers.get(k)}
        headers["x-api-key"] = KEY
        target = UPSTREAM + ("?" + url.query if url.query else "")
        req = urllib.request.Request(target, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=CONFIG.get("timeout_s", 900), context=TLS) as r:
                status, data, rh = r.status, r.read(), r.headers
        except urllib.error.HTTPError as e:
            status, data, rh = e.code, e.read(), e.headers
        except (urllib.error.URLError, OSError):
            return self._error(502, "upstream not reachable", t0, model)
        self._reply(status, data, [(k, rh[k]) for k in RESPONSE_HEADERS if rh.get(k)])
        self._note(status, model, t0)

    def do_GET(self):
        self._error(405, "method not allowed", time.time())

    do_PUT = do_DELETE = do_PATCH = do_HEAD = do_GET


if __name__ == "__main__":
    server = http.server.ThreadingHTTPServer(("127.0.0.1", int(CONFIG["port"])), Handler)
    sys.stderr.write("llmproxy on 127.0.0.1:{}, models {}\n".format(CONFIG["port"], CONFIG["models"]))
    server.serve_forever()

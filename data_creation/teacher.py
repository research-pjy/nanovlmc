"""Explicit, local-only Ollama API; never downloads or starts a server."""
import json
from urllib.parse import urlparse
from urllib.request import Request, build_opener, ProxyHandler


class Ollama:
    def __init__(self, host, timeout=180):
        url = urlparse(host)
        if url.scheme != "http" or url.hostname not in ("127.0.0.1", "localhost", "::1") or url.path not in ("", "/") or url.username or url.query or url.fragment:
            raise ValueError("Use an explicit loopback http://host:port for the job-local server")
        self.host, self.timeout = host.rstrip("/"), timeout
        self.opener = build_opener(ProxyHandler({}))

    def call(self, path, body=None):
        request = Request(self.host + path, data=None if body is None else json.dumps(body).encode(),
                          headers={"Content-Type": "application/json"})
        with self.opener.open(request, timeout=self.timeout) as response:
            return json.load(response)

    def metadata(self, model):
        models = self.call("/api/tags")["models"]
        matches = [m for m in models if m.get("name") == model or m.get("model") == model]
        if len(matches) != 1:
            raise ValueError(f"Exact model tag {model!r} is not uniquely installed; no automatic pull")
        show = self.call("/api/show", {"model": model})
        identity = {"model": model, "digest": matches[0]["digest"], "details": show.get("details", {}),
                    "template": show.get("template"), "parameters": show.get("parameters"),
                    "system": show.get("system"), "model_info": show.get("model_info"),
                    "capabilities": show.get("capabilities"), "server_version": self.call("/api/version")["version"]}
        if not identity["details"].get("quantization_level"):
            raise ValueError("Server did not identify model quantization")
        return identity

    def generate(self, model, prompt, options):
        return self.call("/api/generate", {"model": model, "prompt": prompt,
                        "options": options, "stream": False, "keep_alive": "10m"})

    def residency(self):
        return self.call("/api/ps")

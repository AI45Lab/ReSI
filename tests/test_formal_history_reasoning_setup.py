from __future__ import annotations

import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from r2ai_autoresearch.formal_eval.api import chat_completion
from r2ai_autoresearch.formal_eval.cli import _generation_extra_body
from r2ai_autoresearch.formal_eval.config import Endpoint, GenerationConfig


class Handler(BaseHTTPRequestHandler):
    request_body = None
    reasoning_field = "reasoning_content"

    def log_message(self, format, *args):
        return

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        type(self).request_body = json.loads(self.rfile.read(length))
        body = json.dumps(
            {
                "id": "reasoning-test",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "437",
                        },
                        "finish_reason": "stop",
                    }
                ],
            }
        )
        payload = json.loads(body)
        payload["choices"][0]["message"][type(self).reasoning_field] = "synthetic reasoning"
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class ReasoningSetupTests(unittest.TestCase):
    def test_generation_extra_body_validation(self):
        self.assertEqual(
            _generation_extra_body('{"top_k":20,"min_p":0.0}'),
            {"top_k": 20, "min_p": 0.0},
        )
        with self.assertRaises(ValueError):
            _generation_extra_body('{"temperature":0.7}')

    def test_generation_extra_body_reaches_request(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            endpoint = Endpoint(
                model="mock", base_url=f"http://127.0.0.1:{server.server_port}/v1"
            )
            result = chat_completion(
                endpoint,
                [{"role": "user", "content": "19 * 23"}],
                GenerationConfig(
                    temperature=1.0,
                    top_p=0.95,
                    max_tokens=32,
                    extra_body={"top_k": 20, "min_p": 0.0},
                ),
                retries=0,
            )
            self.assertEqual(result["reasoning_content"], "synthetic reasoning")
            self.assertEqual(Handler.request_body["top_k"], 20)
            self.assertEqual(Handler.request_body["min_p"], 0.0)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_vllm_reasoning_alias_is_normalized(self):
        Handler.reasoning_field = "reasoning"
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            endpoint = Endpoint(
                model="mock", base_url=f"http://127.0.0.1:{server.server_port}/v1"
            )
            result = chat_completion(
                endpoint,
                [{"role": "user", "content": "19 * 23"}],
                GenerationConfig(temperature=1.0, top_p=0.95, max_tokens=32),
                retries=0,
            )
            self.assertEqual(result["reasoning_content"], "synthetic reasoning")
        finally:
            Handler.reasoning_field = "reasoning_content"
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_official_reasoning_system_prompt_is_prepended(self):
        Handler.reasoning_field = "reasoning_content"
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt") as prompt:
                prompt.write("Rules [THINK]scratch[/THINK] answer")
                prompt.flush()
                endpoint = Endpoint(
                    model="mock",
                    base_url=f"http://127.0.0.1:{server.server_port}/v1",
                    default_system_prompt_file=prompt.name,
                )
                chat_completion(
                    endpoint,
                    [{"role": "user", "content": "19 * 23"}],
                    GenerationConfig(temperature=1.0, top_p=0.95, max_tokens=32),
                    retries=0,
                )
                system = Handler.request_body["messages"][0]
                self.assertEqual(system["role"], "system")
                self.assertEqual(system["content"][0]["text"], "Rules ")
                self.assertEqual(system["content"][1]["type"], "thinking")
                self.assertTrue(system["content"][1]["closed"])
                self.assertEqual(system["content"][1]["thinking"], "scratch")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()

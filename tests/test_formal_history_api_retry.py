from __future__ import annotations

import unittest
from unittest.mock import patch

from r2ai_autoresearch.formal_eval.api import chat_completion
from r2ai_autoresearch.formal_eval.config import Endpoint, GenerationConfig


def _response() -> dict:
    return {
        "id": "eventual-success",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "stable final answer",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {},
    }


class APITransientRetryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.endpoint = Endpoint(
            model="mock-model",
            base_url="http://127.0.0.1:1/v1",
        )
        self.generation = GenerationConfig(max_tokens=16)

    def test_vllm_output_cap_overflow_preserves_requested_budget(self) -> None:
        bodies=[]
        def respond(endpoint, route, *, body, timeout):
            bodies.append(dict(body))
            if len(bodies)==1:
                raise RuntimeError('HTTP 400: max_tokens=65536 cannot be greater than max_model_len=max_total_tokens=40960. Please request fewer output tokens.')
            return _response()
        with patch('r2ai_autoresearch.formal_eval.api._request_json', side_effect=respond):
            result=chat_completion(self.endpoint,[{'role':'user','content':'test'}],GenerationConfig(max_tokens=65536))
        self.assertEqual([b['max_tokens'] for b in bodies],[65536,40896])
        self.assertEqual(result['requested_max_tokens'],65536)
        self.assertEqual(result['effective_max_tokens'],40896)
        self.assertTrue(result['context_window_adjusted'])

    def test_endpoint_limits_bound_unavailable_service_requests(self) -> None:
        endpoint = Endpoint.from_mapping({
            "model": "unavailable", "base_url": "http://127.0.0.1:1/v1",
            "timeout_seconds": 7, "max_retries": 0, "transient_retries": 1,
        })
        with patch(
            "r2ai_autoresearch.formal_eval.api._request_json",
            side_effect=TimeoutError("service timed out"),
        ) as request, patch("r2ai_autoresearch.formal_eval.api.time.sleep"):
            with self.assertRaises(TimeoutError):
                chat_completion(endpoint, [{"role": "user", "content": "test"}], self.generation)
        self.assertEqual(request.call_count, 2)
        self.assertTrue(all(call.kwargs["timeout"] == 7 for call in request.call_args_list))

    def test_endpoint_rejects_unbounded_timeout_or_retry_values(self) -> None:
        for override in ({"timeout_seconds": float("inf")}, {"max_retries": -1}, {"transient_retries": 13}):
            with self.subTest(override=override), self.assertRaises(ValueError):
                Endpoint.from_mapping({"model": "fixture", "base_url": "http://localhost:1/v1", **override})

    def test_connection_errors_get_extended_retry_window(self) -> None:
        failures = [
            RuntimeError("File API proxy v2 error: Connection error.")
            for _ in range(5)
        ]
        with patch(
            "r2ai_autoresearch.formal_eval.api._request_json",
            side_effect=[*failures, _response()],
        ) as request, patch("r2ai_autoresearch.formal_eval.api.time.sleep") as sleep:
            result = chat_completion(
                self.endpoint,
                [{"role": "user", "content": "test"}],
                self.generation,
                retries=4,
                transient_retries=8,
                transient_max_delay=0,
            )
        self.assertEqual(result["response"], "stable final answer")
        self.assertEqual(request.call_count, 6)
        self.assertEqual(sleep.call_count, 5)

    def test_non_transient_errors_keep_normal_retry_limit(self) -> None:
        with patch(
            "r2ai_autoresearch.formal_eval.api._request_json",
            side_effect=ValueError("invalid response schema"),
        ) as request, patch("r2ai_autoresearch.formal_eval.api.time.sleep"):
            with self.assertRaisesRegex(ValueError, "invalid response schema"):
                chat_completion(
                    self.endpoint,
                    [{"role": "user", "content": "test"}],
                    self.generation,
                    retries=1,
                    transient_retries=8,
                    transient_max_delay=0,
                )
        self.assertEqual(request.call_count, 2)

    def test_http_5xx_is_transient(self) -> None:
        with patch(
            "r2ai_autoresearch.formal_eval.api._request_json",
            side_effect=[RuntimeError("HTTP 503 from upstream"), _response()],
        ) as request, patch("r2ai_autoresearch.formal_eval.api.time.sleep"):
            result = chat_completion(
                self.endpoint,
                [{"role": "user", "content": "test"}],
                self.generation,
                retries=0,
                transient_retries=1,
                transient_max_delay=0,
            )
        self.assertEqual(result["response"], "stable final answer")
        self.assertEqual(request.call_count, 2)


if __name__ == "__main__":
    unittest.main()

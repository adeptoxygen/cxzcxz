import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import anthropic
import httpx
from fastapi.testclient import TestClient

from adapter import (MAX_BODY, ProtocolError, claude_environment, create_app,
                     from_openai, to_openai, validate_config)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = {**json.loads((ROOT / "config.example.json").read_text()), "auth_token": "test-token-" * 5,
          "upstream_api_key": "upstream-only-token"}
TOOL = {"name": "Read", "description": "Read a file", "input_schema": {
    "type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}


def request(**kw):
    return {"model": "client-alias", "max_tokens": 1000,
            "messages": [{"role": "user", "content": "Привет"}], **kw}


def completion(content="Ответ", calls=None, finish="stop"):
    return {"choices": [{"message": {"content": content, "tool_calls": calls}, "finish_reason": finish}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 4}}


def call(arguments='{"path":"hello.txt"}', name="Read"):
    return {"id": "call_1", "type": "function", "function": {"name": name, "arguments": arguments}}


class ConversionTests(unittest.TestCase):
    def test_system_schema_and_model(self):
        converted = to_openai(request(system=[{"type": "text", "text": "system", "cache_control": {
            "type": "ephemeral"}}], tools=[TOOL]), "hoplite-agent")
        self.assertEqual(converted["messages"][0], {"role": "system", "content": "system"})
        self.assertEqual(converted["tools"][0]["function"]["parameters"], TOOL["input_schema"])
        self.assertEqual(converted["model"], "hoplite-agent")
        self.assertFalse(converted["stream"])

    def test_tool_loop_with_error_and_followup(self):
        history = [
            {"role": "assistant", "content": [{"type": "text", "text": "Reading"},
                {"type": "tool_use", "id": "call_1", "name": "Read", "input": {"path": "x"}}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "call_1", "is_error": True,
                 "content": [{"type": "text", "text": "Not found"}]},
                {"type": "text", "text": "Try another file"}]}]
        messages = to_openai(request(messages=history), "upstream")["messages"]
        self.assertEqual(messages[0]["tool_calls"][0]["id"], "call_1")
        self.assertEqual(messages[1], {"role": "tool", "tool_call_id": "call_1",
                                      "content": "Tool error: Not found"})
        self.assertEqual(messages[2]["content"], "Try another file")

    def test_session_id_not_account_id(self):
        body = request(metadata={"user_id": json.dumps({"user_id": "account", "session_id": "session"})})
        self.assertEqual(to_openai(body, "upstream")["user"], "session")
        self.assertNotIn("user", to_openai(request(metadata={"user_id": "account"}), "upstream"))

    def test_tool_choice(self):
        result = to_openai(request(tools=[TOOL], tool_choice={"type": "tool", "name": "Read",
                              "disable_parallel_tool_use": True}), "upstream")
        self.assertEqual(result["tool_choice"]["function"]["name"], "Read")
        self.assertFalse(result["parallel_tool_calls"])

    def test_effort_from_claude_cli(self):
        result = to_openai(request(output_config={"effort": "high"}), "upstream")
        self.assertEqual(result["reasoning_effort"], "high")
        with self.assertRaises(ProtocolError):
            to_openai(request(output_config={"format": {"type": "json_schema"}}), "upstream")

    def test_inline_system_message_from_claude_cli(self):
        body = request(messages=[{"role": "user", "content": "hello"},
                                 {"role": "system", "content": [{"type": "text", "text": "context"}]}])
        self.assertEqual(to_openai(body, "upstream")["messages"][0],
                         {"role": "system", "content": "context"})

    def test_thinking_hints_do_not_require_native_thinking_support(self):
        for thinking in ({"type": "disabled"}, {"type": "adaptive"},
                         {"type": "enabled", "budget_tokens": 1024}):
            with self.subTest(thinking=thinking):
                result = to_openai(request(thinking=thinking, output_config={"effort": "high"}), "upstream")
                self.assertNotIn("thinking", result)
                self.assertNotIn("budget_tokens", result)
                self.assertEqual(result["reasoning_effort"], "high")

    def test_invalid_thinking_and_signed_history_still_rejected(self):
        for thinking in (None, [], {"type": "unknown"}, {"type": "enabled"},
                         {"type": "enabled", "budget_tokens": True},
                         {"type": "enabled", "budget_tokens": -1}):
            with self.subTest(thinking=thinking), self.assertRaises(ProtocolError):
                to_openai(request(thinking=thinking), "upstream")
        with self.assertRaises(ProtocolError):
            to_openai(request(messages=[{"role": "assistant", "content": [
                {"type": "thinking", "thinking": "private", "signature": "test"}]}]), "upstream")

    def test_unsupported_or_invalid_requests(self):
        for body in [[], request(max_tokens=True), request(stream="yes"), request(messages=[]),
                     request(thinking={"type": "enabled", "budget_tokens": "100"}),
                     request(messages=[{"role": "user", "content": [{"type": "image"}]}]),
                     request(tools=[{"type": "web_search_20250305", "name": "web_search"}]),
                     request(tool_choice={"type": "tool", "name": "absent"})]:
            with self.subTest(body=body), self.assertRaises(ProtocolError):
                to_openai(body, "upstream")

    def test_response_text_and_finish(self):
        for upstream, expected in [("stop", "end_turn"), ("length", "max_tokens"), ("content_filter", "refusal")]:
            result = from_openai(completion(finish=upstream), "alias", set())
            self.assertEqual(result["stop_reason"], expected)
            self.assertEqual(result["model"], "alias")
            self.assertEqual(result["usage"], {"input_tokens": 12, "output_tokens": 4})

    def test_multiple_tools(self):
        second = {**call(), "id": "call_2"}
        result = from_openai(completion(None, [call(), second], "tool_calls"), "alias", {"Read"})
        self.assertEqual(result["stop_reason"], "tool_use")
        self.assertEqual([b["id"] for b in result["content"]], ["call_1", "call_2"])

    def test_malformed_upstream_rejected(self):
        for data in [{}, {"choices": []}, completion(calls=[call("bad")]),
                     completion(calls=[call("[]")]), completion(calls=[call(name="Bash")]),
                     completion(calls=[call(), call()]), completion(finish="tool_calls"),
                     completion(finish=None)]:
            with self.subTest(data=data), self.assertRaises(ProtocolError) as exc:
                from_openai(data, "alias", {"Read"})
            self.assertEqual(exc.exception.status, 502)


class HTTPTests(unittest.TestCase):
    def setUp(self):
        self.seen = []
        self.response = httpx.Response(200, json=completion())

        def upstream(req):
            self.seen.append(req)
            return self.response

        self.client = TestClient(create_app(CONFIG, httpx.MockTransport(upstream)))
        self.headers = {"Authorization": "Bearer " + CONFIG["auth_token"]}

    def tearDown(self):
        self.client.close()

    def post(self, body=None, **kw):
        return self.client.post("/v1/messages", json=body or request(), headers=self.headers, **kw)

    def test_auth_and_liveness(self):
        self.assertEqual(self.client.get("/health").json(), {"status": "ok", "upstream_checked": False})
        self.assertEqual(self.client.post("/v1/messages", json=request()).status_code, 401)
        self.assertEqual(self.client.get("/v1/models", headers={"x-api-key": CONFIG["auth_token"]}).status_code, 200)
        self.assertEqual(self.seen, [])

    def test_nonstream_and_separate_credentials(self):
        response = self.post()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["content"][0]["text"], "Ответ")
        self.assertEqual(self.seen[0].headers["authorization"], "Bearer upstream-only-token")
        self.assertNotIn(CONFIG["auth_token"], str(self.seen[0].headers))
        self.assertEqual(str(self.seen[0].url), "http://127.0.0.1:8787/v1/chat/completions")

    def test_sse_parsed_by_official_sdk(self):
        self.response = httpx.Response(200, json=completion("Reading", [call()], "tool_calls"))
        sdk = anthropic.Anthropic(api_key=CONFIG["auth_token"], base_url="http://testserver",
                                 http_client=self.client, max_retries=0)
        with sdk.messages.stream(**request(tools=[TOOL])) as stream:
            message = stream.get_final_message()
        self.assertEqual(message.stop_reason, "tool_use")
        self.assertEqual(message.content[0].text, "Reading")
        self.assertEqual(message.content[1].input, {"path": "hello.txt"})
        self.assertEqual(message.usage.output_tokens, 4)

    def test_thinking_requests_stream_text_and_tools(self):
        for thinking in ({"type": "adaptive"}, {"type": "enabled", "budget_tokens": 1024}):
            with self.subTest(thinking=thinking):
                self.response = httpx.Response(200, json=completion("Reading", [call()], "tool_calls"))
                response = self.post(request(thinking=thinking, stream=True, tools=[TOOL]))
                self.assertEqual(response.status_code, 200)
                self.assertIn('"type": "tool_use"', response.text)
                self.assertIn('"type": "text_delta"', response.text)
                self.assertIn("event: message_stop", response.text)
                self.assertNotIn("thinking_delta", response.text)
                self.assertNotIn("thinking", json.loads(self.seen[-1].content))

    def test_upstream_error_redacted_json_and_sse(self):
        for status in (401, 403, 429, 500, 503, 504):
            self.response = httpx.Response(status, text="secret internal response")
            for stream in (False, True):
                with self.subTest(status=status, stream=stream):
                    result = self.post(request(stream=stream))
                    self.assertNotIn("secret internal", result.text)
                    if stream:
                        self.assertIn("event: error", result.text)
                        self.assertNotIn("event: message_stop", result.text)
                    else:
                        self.assertEqual(result.status_code, 502 if status == 500 else status)
                        self.assertEqual(result.json()["type"], "error")

    def test_stream_error_sdk_raises(self):
        self.response = httpx.Response(429, text="private")
        sdk = anthropic.Anthropic(api_key=CONFIG["auth_token"], base_url="http://testserver",
                                 http_client=self.client, max_retries=0)
        with self.assertRaises(anthropic.APIError):
            with sdk.messages.stream(**request()) as stream:
                stream.get_final_message()

    def test_invalid_json_and_body_limit(self):
        for data, status in [("not json", 400), ('"not object"', 400), ("x" * (MAX_BODY + 1), 413)]:
            result = self.client.post("/v1/messages", content=data, headers=self.headers)
            self.assertEqual(result.status_code, status)
        self.assertEqual(self.seen, [])

    def test_count_tokens_not_fabricated(self):
        self.assertEqual(self.client.post("/v1/messages/count_tokens", headers=self.headers).status_code, 501)

    def test_network_errors(self):
        for error, status in [(httpx.ConnectError("private"), 502), (httpx.ReadTimeout("private"), 504)]:
            def fail(req):
                raise error
            with TestClient(create_app(CONFIG, httpx.MockTransport(fail))) as client:
                result = client.post("/v1/messages", json=request(), headers=self.headers)
                self.assertEqual(result.status_code, status)
                self.assertNotIn("private", result.text)


class SetupTests(unittest.TestCase):
    def test_init_never_overwrites(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            command = [sys.executable, str(ROOT / "adapter.py"), "--config", str(path), "init"]
            first = subprocess.run(command, capture_output=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            content = path.read_text()
            validate_config(json.loads(content))
            second = subprocess.run(command, capture_output=True)
            self.assertNotEqual(second.returncode, 0)
            self.assertEqual(content, path.read_text())
            self.assertNotIn(json.loads(content)["auth_token"].encode(), first.stdout + first.stderr)
            if os.name != "nt":
                self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_insecure_config_rejected(self):
        for changes in [{"auth_token": "short"}, {"host": "0.0.0.0"},
                        {"upstream_url": "http://192.168.1.1/v1"},
                        {"upstream_url": "https://user:pass@example.com/v1"},
                        {"timeout_seconds": -1}]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_config({**CONFIG, **changes})

    def test_launcher_environment_is_scoped(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "unrelated", "CLAUDE_CODE_USE_BEDROCK": "1"}):
            env = claude_environment(CONFIG)
            self.assertNotIn("ANTHROPIC_API_KEY", env)
            self.assertNotIn("CLAUDE_CODE_USE_BEDROCK", env)
            self.assertEqual(env["ANTHROPIC_AUTH_TOKEN"], CONFIG["auth_token"])
            self.assertEqual(env["ANTHROPIC_BASE_URL"], "http://127.0.0.1:8788")
            self.assertEqual(os.environ["ANTHROPIC_API_KEY"], "unrelated")


if __name__ == "__main__":
    unittest.main()

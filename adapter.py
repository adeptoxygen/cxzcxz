"""Experimental Anthropic Messages facade for a local Hoplite OpenAI gateway."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hmac
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
from urllib.parse import urlsplit
import uuid

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

from client import claude_environment as remote_environment

ROOT = Path(__file__).resolve().parent
MAX_BODY = 8 * 1024 * 1024


class ProtocolError(Exception):
    def __init__(self, message, status=400, kind="invalid_request_error"):
        self.message, self.status, self.kind = message, status, kind

    def body(self):
        return {"type": "error", "error": {"type": self.kind, "message": self.message}}


def require(condition, message):
    if not condition:
        raise ProtocolError(message)


def text_content(value):
    if isinstance(value, str):
        return value
    require(isinstance(value, list), "Content must be a string or block array.")
    texts = []
    for block in value:
        require(isinstance(block, dict) and block.get("type") == "text",
                "Only text content is supported here; images, documents and thinking are unavailable.")
        require(isinstance(block.get("text"), str), "Text blocks require text.")
        texts.append(block["text"])
    return "\n".join(texts)


def to_openai(body, model):
    require(isinstance(body, dict), "Expected a JSON object.")
    require(isinstance(body.get("model"), str) and body["model"], "model is required.")
    require(type(body.get("max_tokens")) is int and body["max_tokens"] > 0,
            "max_tokens must be a positive integer.")
    require(type(body.get("stream", False)) is bool, "stream must be a boolean.")
    thinking = body.get("thinking", {"type": "disabled"})
    require(isinstance(thinking, dict) and thinking.get("type") in ("disabled", "enabled", "adaptive"),
            "thinking.type must be disabled, enabled or adaptive.")
    if thinking["type"] == "enabled":
        require(type(thinking.get("budget_tokens")) is int and thinking["budget_tokens"] > 0,
                "Enabled thinking requires a positive integer budget_tokens.")
    # Claude Code may enable thinking despite launcher flags. The gateway cannot
    # return signed thinking blocks or enforce token budgets; accept the hint only.
    for key in ("context_management", "container", "mcp_servers"):
        require(not body.get(key), f"{key} is unsupported by this experimental gateway.")
    messages = []
    if body.get("system"):
        messages.append({"role": "system", "content": text_content(body["system"])})
    history = body.get("messages")
    require(isinstance(history, list) and history, "messages must be a nonempty array.")
    for message in history:
        require(isinstance(message, dict) and message.get("role") in ("user", "assistant", "system"),
                "Message role must be user, assistant or system.")
        role = message["role"]
        content = message.get("content")
        if role == "system":
            messages.append({"role": role, "content": text_content(content)})
            continue
        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue
        require(isinstance(content, list), "Message content must be a string or block array.")
        texts, calls, results = [], [], []
        for block in content:
            require(isinstance(block, dict), "Invalid content block.")
            kind = block.get("type")
            if kind == "text":
                texts.append(text_content([block]))
            elif kind == "tool_use" and role == "assistant":
                require(all(isinstance(block.get(k), str) and block[k] for k in ("id", "name"))
                        and isinstance(block.get("input"), dict), "Invalid tool_use block.")
                calls.append({"id": block["id"], "type": "function", "function": {
                    "name": block["name"], "arguments": json.dumps(block["input"], ensure_ascii=False)}})
            elif kind == "tool_result" and role == "user":
                require(isinstance(block.get("tool_use_id"), str) and block["tool_use_id"],
                        "tool_result requires tool_use_id.")
                result = text_content(block.get("content", ""))
                if block.get("is_error"):
                    result = "Tool error: " + result
                results.append({"role": "tool", "tool_call_id": block["tool_use_id"], "content": result})
            else:
                raise ProtocolError(f"Unsupported content block: {kind!s}.")
        # OpenAI requires tool results immediately after the assistant tool calls.
        messages.extend(results)
        if texts or calls or not results:
            converted = {"role": role, "content": "\n".join(texts) if texts else None}
            if calls:
                converted["tool_calls"] = calls
            messages.append(converted)
    # hoplite-gateway reads only trailing tool messages for a tool-result turn.
    messages = ([m for m in messages if m["role"] == "system"]
                + [m for m in messages if m["role"] != "system"])
    result = {"model": model, "messages": messages, "max_tokens": body["max_tokens"], "stream": False}
    output = body.get("output_config", {})
    require(isinstance(output, dict) and not (set(output) - {"effort"}),
            "Only output_config.effort is supported; structured output is unavailable.")
    if "effort" in output:
        require(output["effort"] in ("low", "medium", "high", "xhigh", "max"), "Unsupported effort.")
        result["reasoning_effort"] = output["effort"]
    tools = body.get("tools", [])
    require(isinstance(tools, list), "tools must be an array.")
    names = set()
    if tools:
        converted = []
        for tool in tools:
            require(isinstance(tool, dict) and tool.get("type", "custom") == "custom",
                    "Only client-side custom tools are supported.")
            require(isinstance(tool.get("name"), str) and tool["name"]
                    and isinstance(tool.get("input_schema"), dict), "Invalid tool definition.")
            require(tool["name"] not in names, "Duplicate tool name.")
            names.add(tool["name"])
            converted.append({"type": "function", "function": {
                "name": tool["name"], "description": tool.get("description", ""),
                "parameters": tool["input_schema"]}})
        result["tools"] = converted
    if "tool_choice" in body:
        choice = body["tool_choice"]
        require(isinstance(choice, dict), "Invalid tool_choice.")
        kind = choice.get("type")
        require(kind in ("auto", "any", "none", "tool"), "Unsupported tool_choice.")
        if kind == "tool":
            require(choice.get("name") in names, "Selected tool is not defined.")
            result["tool_choice"] = {"type": "function", "function": {"name": choice["name"]}}
        else:
            result["tool_choice"] = {"auto": "auto", "any": "required", "none": "none"}[kind]
        if "disable_parallel_tool_use" in choice:
            require(type(choice["disable_parallel_tool_use"]) is bool, "Invalid parallel tool setting.")
            result["parallel_tool_calls"] = not choice["disable_parallel_tool_use"]
    for key in ("temperature", "top_p", "top_k"):
        if key in body:
            result[key] = body[key]
    if "stop_sequences" in body:
        result["stop"] = body["stop_sequences"]
    # Use Claude's conversation identifier, never its account identifier.
    session = body.get("metadata", {}).get("user_id") if isinstance(body.get("metadata", {}), dict) else None
    if session:
        try:
            session = json.loads(session).get("session_id")
        except (ValueError, TypeError, AttributeError):
            session = None
        if isinstance(session, str) and session:
            result["user"] = session
    return result


def from_openai(data, model, tools):
    try:
        choice = data["choices"][0]
        message = choice["message"]
        content = []
        if message.get("content"):
            if not isinstance(message["content"], str):
                raise ValueError()
            content.append({"type": "text", "text": message["content"]})
        ids = set()
        for call in message.get("tool_calls") or []:
            fn = call["function"]
            arguments = json.loads(fn["arguments"])
            if not isinstance(arguments, dict) or fn["name"] not in tools:
                raise ValueError()
            if not isinstance(call["id"], str) or not call["id"] or call["id"] in ids:
                raise ValueError()
            ids.add(call["id"])
            content.append({"type": "tool_use", "id": call["id"], "name": fn["name"], "input": arguments})
        finish = choice.get("finish_reason")
        if finish not in ("stop", "length", "tool_calls", "content_filter"):
            raise ValueError()
        if finish == "tool_calls" and not ids:
            raise ValueError()
        stop = "tool_use" if ids else {"stop": "end_turn", "length": "max_tokens",
                                      "content_filter": "refusal"}[finish]
        usage = data.get("usage") or {}
        counts = [usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)]
        if any(type(n) is not int or n < 0 for n in counts):
            raise ValueError()
        return {"id": "msg_" + uuid.uuid4().hex, "type": "message", "role": "assistant",
                "model": model, "content": content, "stop_reason": stop, "stop_sequence": None,
                "usage": {"input_tokens": counts[0], "output_tokens": counts[1]}}
    except (KeyError, IndexError, TypeError, ValueError, AttributeError):
        raise ProtocolError("Upstream returned an invalid completion or tool call.", 502, "api_error") from None


def event(kind, **fields):
    return f"event: {kind}\ndata: {json.dumps({'type': kind, **fields}, ensure_ascii=False)}\n\n"


def message_events(message):
    start = {**message, "content": [], "stop_reason": None,
             "usage": {**message["usage"], "output_tokens": 0}}
    yield event("message_start", message=start)
    for index, block in enumerate(message["content"]):
        if block["type"] == "text":
            initial = {"type": "text", "text": ""}
            delta = {"type": "text_delta", "text": block["text"]}
        else:
            initial = {**block, "input": {}}
            delta = {"type": "input_json_delta", "partial_json": json.dumps(block["input"], ensure_ascii=False)}
        yield event("content_block_start", index=index, content_block=initial)
        yield event("content_block_delta", index=index, delta=delta)
        yield event("content_block_stop", index=index)
    yield event("message_delta", delta={"stop_reason": message["stop_reason"], "stop_sequence": None},
                usage={"output_tokens": message["usage"]["output_tokens"]})
    yield event("message_stop")


def validate_config(config):
    if not isinstance(config.get("auth_token"), str) or len(config["auth_token"]) < 32:
        raise ValueError("Run 'python adapter.py init' to generate a local auth_token (32+ characters).")
    url = urlsplit(config["upstream_url"])
    if (url.scheme not in ("http", "https") or not url.hostname or url.username or url.password
            or url.query or url.fragment):
        raise ValueError("upstream_url must be an HTTP(S) base URL without credentials/query/fragment.")
    if url.scheme == "http" and url.hostname not in ("localhost", "127.0.0.1", "::1"):
        raise ValueError("Remote upstreams require HTTPS.")
    if config["host"] != "127.0.0.1":
        raise ValueError("This local adapter must bind to 127.0.0.1; use an authenticated HTTPS tunnel if needed.")
    if type(config["port"]) is not int or not 1 <= config["port"] <= 65535:
        raise ValueError("Invalid port.")
    if not isinstance(config["model"], str) or not config["model"]:
        raise ValueError("Set an explicit upstream model or hoplite-agent.")
    if not isinstance(config["timeout_seconds"], (int, float)) or not 0 < config["timeout_seconds"] <= 3600:
        raise ValueError("timeout_seconds must be between 0 and 3600.")


def create_app(config, transport=None):
    validate_config(config)
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @app.exception_handler(ProtocolError)
    async def protocol_error(request, exc):
        return JSONResponse(exc.body(), status_code=exc.status)

    @app.middleware("http")
    async def authenticate(request, call_next):
        if request.url.path != "/health":
            token = request.headers.get("x-api-key", "")
            auth = request.headers.get("authorization", "")
            if auth.lower().startswith("bearer "):
                token = auth[7:]
            if not hmac.compare_digest(token.encode(), config["auth_token"].encode()):
                error = ProtocolError("Invalid adapter token.", 401, "authentication_error")
                return JSONResponse(error.body(), status_code=401)
        return await call_next(request)

    @app.get("/health")
    async def health():
        return {"status": "ok", "upstream_checked": False}

    @app.get("/v1/models")
    async def models():
        return {"data": [{"id": config["model"], "type": "model", "display_name": config["model"],
                          "created_at": "1970-01-01T00:00:00Z"}], "has_more": False,
                "first_id": config["model"], "last_id": config["model"]}

    @app.post("/v1/messages/count_tokens")
    async def count_tokens():
        raise ProtocolError("This upstream does not provide accurate token counting.", 501, "api_error")

    async def complete(payload, requested_model):
        headers = {}
        if config.get("upstream_api_key"):
            headers["Authorization"] = "Bearer " + config["upstream_api_key"]
        try:
            async with httpx.AsyncClient(transport=transport, timeout=config["timeout_seconds"],
                                         trust_env=False) as client:
                response = await client.post(config["upstream_url"].rstrip("/") + "/chat/completions",
                                             json=payload, headers=headers)
            if response.status_code != 200:
                status = response.status_code if response.status_code in (401, 403, 429, 503, 504) else 502
                kind = {401: "authentication_error", 403: "permission_error", 429: "rate_limit_error",
                        503: "overloaded_error"}.get(status, "api_error")
                raise ProtocolError(f"Upstream HTTP {response.status_code}; check the gateway locally.", status, kind)
            try:
                data = response.json()
            except ValueError:
                raise ProtocolError("Upstream returned invalid JSON.", 502, "api_error") from None
            names = {tool["function"]["name"] for tool in payload.get("tools", [])}
            return from_openai(data, requested_model, names)
        except httpx.TimeoutException:
            raise ProtocolError("Upstream timed out; its cloud run may still be active.", 504, "api_error") from None
        except httpx.HTTPError:
            raise ProtocolError("Cannot reach upstream; start hoplite-gateway and check its address.", 502, "api_error") from None

    @app.post("/v1/messages")
    async def messages(request: Request):
        raw = bytearray()
        async for chunk in request.stream():
            raw.extend(chunk)
            if len(raw) > MAX_BODY:
                raise ProtocolError("Request body exceeds 8 MiB.", 413, "request_too_large")
        try:
            body = json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            raise ProtocolError("Invalid JSON request.") from None
        payload = to_openai(body, config["model"])
        if not body.get("stream"):
            return await complete(payload, body["model"])

        async def stream():
            task = asyncio.create_task(complete(payload, body["model"]))
            try:
                while not task.done():
                    yield event("ping")
                    await asyncio.wait({task}, timeout=10)
                for item in message_events(await task):
                    yield item
            except ProtocolError as exc:
                yield event("error", error=exc.body()["error"])
            finally:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, ProtocolError):
                    await task

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    return app


def claude_environment(config):
    return remote_environment(f"http://127.0.0.1:{config['port']}", config["auth_token"],
                              config["model"], config["timeout_seconds"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("command", choices=("init", "serve", "claude"))
    args, cli_args = parser.parse_known_args()
    if args.command != "claude" and cli_args:
        parser.error("Unexpected arguments.")
    try:
        if args.command == "init":
            config = json.loads((ROOT / "config.example.json").read_text(encoding="utf-8"))
            config["auth_token"] = secrets.token_urlsafe(32)
            fd = os.open(args.config, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as file:
                json.dump(config, file, indent=2)
                file.write("\n")
            print("Local config created. Configure upstream_api_key, then run 'python adapter.py serve'.")
            return
        config = json.loads(args.config.read_text(encoding="utf-8"))
        validate_config(config)
        if args.command == "serve":
            uvicorn.run(create_app(config), host=config["host"], port=config["port"], access_log=False)
        else:
            executable = shutil.which("claude")
            if not executable:
                parser.error("Claude Code CLI is not installed or is missing from PATH.")
            if cli_args[:1] == ["--"]:
                cli_args = cli_args[1:]
            raise SystemExit(subprocess.call([executable, *cli_args], env=claude_environment(config)))
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Configuration/startup error ({type(exc).__name__}). Check --config or run init; existing files are never overwritten.\n")


if __name__ == "__main__":
    main()

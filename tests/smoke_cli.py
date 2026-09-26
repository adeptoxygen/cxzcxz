"""Opt-in real Claude Code smoke test; uses only a mock upstream and a temp file."""

import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time

import httpx
import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adapter import create_app


def main():
    if not shutil.which("claude"):
        raise SystemExit("Install Claude Code first.")
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="adapter-smoke-") as directory:
        path = Path(directory)
        fixture = path / "fixture.txt"
        marker = "LOCAL_TOOL_ROUNDTRIP_73519"
        fixture.write_text(marker, encoding="utf-8")
        seen = []

        def upstream(req):
            body = json.loads(req.content)
            seen.append(body)
            results = [m for m in body["messages"] if m["role"] == "tool"]
            if results:
                assert body["messages"][-1]["role"] == "tool", "Gateway requires trailing tool results"
                assert any(marker in m["content"] for m in results), "Local tool result was lost"
                message = {"content": "LOCAL_ROUNDTRIP_OK"}
                finish = "stop"
            else:
                assert "Read" in [t["function"]["name"] for t in body.get("tools", [])]
                message = {"content": None, "tool_calls": [{"id": "call_smoke", "type": "function",
                    "function": {"name": "Read", "arguments": json.dumps({"file_path": str(fixture)})}}]}
                finish = "tool_calls"
            return httpx.Response(200, json={"choices": [{"message": message, "finish_reason": finish}],
                                            "usage": {"prompt_tokens": 10, "completion_tokens": 5}})

        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        config = {**json.loads((root / "config.example.json").read_text()),
                  "port": sock.getsockname()[1], "auth_token": "smoke-local-token-" * 3}
        config_path = path / "config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        client_path = path / "client.json"
        client_path.write_text(json.dumps({"url": f"http://127.0.0.1:{config['port']}",
                                           "token": config["auth_token"], "model": config["model"]}))
        app = create_app(config, httpx.MockTransport(upstream))
        shapes = []

        @app.middleware("http")
        async def inspect_shape(req, call_next):
            if req.url.path == "/v1/messages":
                body = await req.json()
                shapes.append({"keys": list(body), "roles": [m.get("role") for m in body.get("messages", [])]})
            return await call_next(req)

        server = uvicorn.Server(uvicorn.Config(app, log_level="error", access_log=False))
        thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
        thread.start()
        env = os.environ.copy()
        env.pop("CLAUDECODE", None)
        env["CLAUDE_CONFIG_DIR"] = str(path / "claude-config")
        try:
            deadline = time.monotonic() + 10
            while not server.started:
                if time.monotonic() >= deadline:
                    raise RuntimeError("Test server did not start")
                time.sleep(0.05)
            launcher = ([sys.executable, str(root / "client.py"), "--config", str(client_path), "run", "--"]
                        if "--remote-client" in sys.argv else
                        [sys.executable, str(root / "adapter.py"), "--config", str(config_path), "claude", "--"])
            result = subprocess.run(
                [*launcher,
                 "-p", "Read fixture.txt using Read and report the result.", "--tools", "Read",
                 "--allowedTools", "Read", "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
                 "--setting-sources", "", "--no-session-persistence", "--max-turns", "3",
                 "--output-format", "json"],
                cwd=path, env=env, capture_output=True, text=True, timeout=90)
            if result.returncode or "LOCAL_ROUNDTRIP_OK" not in result.stdout:
                print(result.stdout)
                print(result.stderr, file=sys.stderr)
                print("Request shapes:", shapes)
                raise AssertionError(f"Claude smoke failed; upstream requests: {len(seen)}")
            assert len(seen) >= 2, "Tool roundtrip did not happen"
            print("PASS: real Claude Code -> adapter -> mock gateway -> local Read -> tool_result -> final answer")
        finally:
            server.should_exit = True
            thread.join(timeout=10)
            sock.close()


if __name__ == "__main__":
    main()

"""Manage the temporary gateway, adapter and ngrok tunnel in this workspace."""

import argparse
import io
import json
import os
from pathlib import Path
import re
import secrets
import signal
import subprocess
import sys
import tarfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / ".hoplite" / "runtime"


def private_json(path, data):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as file:
        json.dump(data, file)


def setup():
    RUNTIME.mkdir(parents=True, exist_ok=True)
    subprocess.run(["uv", "venv", "--allow-existing", str(ROOT / ".venv")], check=True)
    subprocess.run(["uv", "pip", "install", "--python", str(ROOT / ".venv/bin/python"),
                    "-r", str(ROOT / "requirements.txt")], check=True)
    gateway = RUNTIME / "gateway"
    gateway.mkdir(exist_ok=True)
    if not (gateway / "server.py").exists():
        url = "https://raw.githubusercontent.com/nikita4a/hoplite-gateway/main/server.py"
        (gateway / "server.py").write_bytes(urllib.request.urlopen(url, timeout=30).read())
    binary = RUNTIME / "ngrok"
    if not binary.exists():
        url = "https://bin.ngrok.com/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz"
        archive = urllib.request.urlopen(url, timeout=60).read()
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
            member = tar.getmember("ngrok")
            if not member.isfile():
                raise ValueError("Invalid ngrok archive")
            binary.write_bytes(tar.extractfile(member).read())
        binary.chmod(0o700)
    subprocess.run([str(binary), "version"], check=True)
    print("Hosting dependencies ready. Private settings are not modified.")


def run():
    os.umask(0o077)
    settings = json.loads((RUNTIME / "hosting-secrets.json").read_text())
    if not all(settings.get(k) for k in ("hoplite_api_key", "ngrok_authtoken", "project_id")):
        raise ValueError("Missing private hosting settings")
    adapter_path = RUNTIME / "adapter-config.json"
    adapter = json.loads(adapter_path.read_text()) if adapter_path.exists() else {
        **json.loads((ROOT / "config.example.json").read_text()),
        "auth_token": secrets.token_urlsafe(32), "upstream_api_key": secrets.token_urlsafe(32),
        "timeout_seconds": 600,
    }
    private_json(adapter_path, adapter)
    private_json(RUNTIME / "gateway/config.json", {
        "api_key": settings["hoplite_api_key"], "api_base": "https://api.hoplite.sh",
        "project_id": settings["project_id"], "gateway_key": adapter["upstream_api_key"], "deadline_s": 540,
    })
    # ngrok accepts JSON as YAML; request inspection is disabled at tunnel startup.
    private_json(RUNTIME / "ngrok.json", {"version": "2", "authtoken": settings["ngrok_authtoken"],
                                          "web_addr": "127.0.0.1:4040"})
    commands = [
        ("gateway", [sys.executable, str(RUNTIME / "gateway/server.py")]),
        ("adapter", [sys.executable, str(ROOT / "adapter.py"), "--config", str(adapter_path), "serve"]),
        ("ngrok", [str(RUNTIME / "ngrok"), "http", "http://127.0.0.1:8788", "--inspect=false",
                   "--config", str(RUNTIME / "ngrok.json"), "--log", "stdout", "--log-format", "json"]),
    ]
    if settings.get("ngrok_url"):
        from client import validate_url
        url = validate_url(settings["ngrok_url"])
        if not url.startswith("https://"):
            raise ValueError("Public tunnel URL must use HTTPS")
        commands[-1][1].extend(["--url", url])
    processes, logs = [], []

    def stop(signum, frame):
        raise KeyboardInterrupt()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        for name, command in commands:
            log = open(RUNTIME / f"{name}.log", "w")
            logs.append(log)
            processes.append((name, subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=log)))
        print("Temporary hosted bridge started; private logs are in runtime storage.", flush=True)
        while True:
            for name, process in processes:
                if process.poll() is not None:
                    codes = sorted(set(re.findall(r"ERR_NGROK_\d+", (RUNTIME / f"{name}.log").read_text())))
                    raise RuntimeError(f"{name} exited ({process.returncode}); {' '.join(codes)}; inspect private logs")
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        for _, process in processes:
            if process.poll() is None:
                process.terminate()
        for _, process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for log in logs:
            log.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("setup", "run"))
    args = parser.parse_args()
    try:
        (setup if args.command == "setup" else run)()
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f"Hosting failed: {type(error).__name__}", file=sys.stderr)
        if isinstance(error, RuntimeError):
            print(str(error), file=sys.stderr)
        sys.exit(1)

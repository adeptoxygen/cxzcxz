"""Linux client for a remotely hosted adapter. Requires only Python and Claude Code."""

import argparse
import getpass
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from urllib.parse import urlsplit


def validate_url(value):
    url = urlsplit(value)
    local = url.scheme == "http" and url.hostname in ("127.0.0.1", "localhost", "::1")
    if (not (url.scheme == "https" or local) or not url.hostname or url.username or url.password
            or url.query or url.fragment or url.path not in ("", "/")):
        raise ValueError("Use an HTTPS server address without /v1, credentials, query or fragment.")
    return value.rstrip("/")


def claude_environment(base_url, token, model, timeout_seconds=300):
    env = os.environ.copy()
    for name in ("ANTHROPIC_API_KEY", "CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX",
                 "CLAUDE_CODE_USE_FOUNDRY"):
        env.pop(name, None)
    env.update(ANTHROPIC_BASE_URL=base_url, ANTHROPIC_AUTH_TOKEN=token, ANTHROPIC_MODEL=model,
               ANTHROPIC_DEFAULT_OPUS_MODEL=model, ANTHROPIC_DEFAULT_SONNET_MODEL=model,
               ANTHROPIC_DEFAULT_HAIKU_MODEL=model, ANTHROPIC_SMALL_FAST_MODEL=model,
               MAX_THINKING_TOKENS="0", CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING="1",
               CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1", ENABLE_TOOL_SEARCH="false",
               API_TIMEOUT_MS=str(int(timeout_seconds * 1000 + 30000)))
    return env


def save_config(path, config):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, temporary = tempfile.mkstemp(prefix=".client-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(config, file, indent=2)
            file.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def validate_config(config):
    validate_url(config["url"])
    if not isinstance(config["token"], str) or len(config["token"]) < 32:
        raise ValueError("Adapter token must contain at least 32 characters, not your Hoplite API key.")
    if not isinstance(config["model"], str) or not config["model"]:
        raise ValueError("Model is required.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default = Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config")))
    parser.add_argument("--config", type=Path, default=default / "hoplite-claude" / "client.json")
    parser.add_argument("command", choices=("configure", "run"))
    args, cli_args = parser.parse_known_args()
    try:
        if args.command == "configure":
            if cli_args:
                parser.error("configure does not accept credentials as arguments.")
            config = {"url": validate_url(input("Adapter HTTPS URL: ").strip()),
                      "token": getpass.getpass("Adapter token (hidden, not Hoplite API key): ").strip(),
                      "model": input("Gateway model [hoplite-agent]: ").strip() or "hoplite-agent"}
            validate_config(config)
            save_config(args.config, config)
            print("Saved private client configuration. Run: python3 client.py run")
            return
        config = json.loads(args.config.read_text(encoding="utf-8"))
        validate_config(config)
        executable = shutil.which("claude")
        if not executable:
            parser.error("Install Claude Code first: https://code.claude.com/docs/en/setup")
        if cli_args[:1] == ["--"]:
            cli_args = cli_args[1:]
        raise SystemExit(subprocess.call([executable, *cli_args], env=claude_environment(
            config["url"], config["token"], config["model"])))
    except (OSError, ValueError, KeyError, TypeError, EOFError):
        parser.exit(1, "Client configuration/startup failed. Run configure with an HTTPS adapter URL and its token.\n")


if __name__ == "__main__":
    main()

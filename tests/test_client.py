import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from client import save_config, validate_config, validate_url


class ClientTests(unittest.TestCase):
    def test_remote_requires_https(self):
        self.assertEqual(validate_url("https://example.com/"), "https://example.com")
        self.assertEqual(validate_url("http://127.0.0.1:8788"), "http://127.0.0.1:8788")
        for value in ("http://example.com", "https://example.com/v1", "https://user@example.com",
                      "https://example.com/?key=value", "https://example.com/#fragment", "file:///x"):
            with self.subTest(url=value), self.assertRaises(ValueError):
                validate_url(value)

    def test_private_atomic_config(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config" / "client.json"
            config = {"url": "https://example.com", "token": "local-test-token-" * 3, "model": "hoplite-agent"}
            validate_config(config)
            save_config(path, config)
            self.assertEqual(json.loads(path.read_text()), config)
            config["model"] = "another-model"
            save_config(path, config)
            self.assertEqual(json.loads(path.read_text()), config)
            self.assertEqual(list(path.parent.iterdir()), [path])
            if os.name != "nt":
                self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_launch_no_dependencies_and_keeps_cwd(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            executable = path / "claude"
            executable.write_text(
                '#!/bin/sh\n'
                'test "$ANTHROPIC_BASE_URL" = "https://example.com" || exit 91\n'
                'test "$ANTHROPIC_MODEL" = "hoplite-agent" || exit 92\n'
                'test "$ANTHROPIC_AUTH_TOKEN" = "client-test-token-client-test-token" || exit 93\n'
                'test "$2" = "hello world" || exit 94\n'
                'pwd\nexit 7\n')
            executable.chmod(0o700)
            config_path = path / "client.json"
            save_config(config_path, {"url": "https://example.com", "token": "client-test-token-client-test-token",
                                     "model": "hoplite-agent"})
            result = subprocess.run([sys.executable, "-S", str(root / "client.py"), "--config", str(config_path),
                                     "run", "--", "-p", "hello world"], cwd=path, capture_output=True, text=True,
                                    env={**os.environ, "PATH": str(path) + os.pathsep + os.environ["PATH"]})
            self.assertEqual(result.returncode, 7, result.stderr)
            self.assertEqual(result.stdout.strip(), str(path))
            self.assertNotIn("client-test-token", result.stdout + result.stderr)

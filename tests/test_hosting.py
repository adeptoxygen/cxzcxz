import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import hosting


class HostingTests(unittest.TestCase):
    def test_private_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "private.json"
            hosting.private_json(path, {"token": "test-only"})
            self.assertEqual(json.loads(path.read_text()), {"token": "test-only"})
            if os.name != "nt":
                self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_missing_settings_do_not_launch_processes(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            (runtime / "hosting-secrets.json").write_text('{}')
            with patch.object(hosting, "RUNTIME", runtime), patch.object(hosting.subprocess, "Popen") as launch:
                mask = os.umask(0o077)
                try:
                    with self.assertRaises(ValueError):
                        hosting.run()
                    launch.assert_not_called()
                finally:
                    os.umask(mask)

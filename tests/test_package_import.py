import subprocess
import sys
import unittest
from pathlib import Path


class PackageImportTest(unittest.TestCase):
    def test_current_namespace_imports_without_deprecation_warning(self):
        src = Path(__file__).resolve().parents[1] / "src"
        subprocess.run(
            [sys.executable, "-B", "-W", "error::DeprecationWarning", "-c",
             f"import sys; sys.path.insert(0, {str(src)!r}); import data_synthetic"],
            check=True,
            capture_output=True,
            text=True,
        )


if __name__ == "__main__":
    unittest.main()

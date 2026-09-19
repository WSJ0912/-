from __future__ import annotations

import json
import unittest
from pathlib import Path

from cxr_research.labels import CHEXPERT_LABELS


class ContractSchemaTests(unittest.TestCase):
    def test_schema_files_are_valid_json_and_model_schema_has_fixed_labels(self) -> None:
        root = Path(__file__).resolve().parents[2] / "contracts" / "schemas"
        for path in root.glob("*.json"):
            with self.subTest(path=path.name):
                schema = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(schema["$schema"], "https://json-schema.org/draft/2020-12/schema")
        manifest = json.loads((root / "model-manifest.schema.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["properties"]["labels"]["const"], list(CHEXPERT_LABELS))


if __name__ == "__main__":
    unittest.main()

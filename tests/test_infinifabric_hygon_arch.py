import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "infinifabric-build-v1"
    / "inspect_hygon_arch.py"
)


def load_module():
    if not MODULE_PATH.is_file():
        raise AssertionError(f"production module is missing: {MODULE_PATH}")
    spec = importlib.util.spec_from_file_location("inspect_hygon_arch", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError("cannot create module specification")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class HygonArchitectureTests(unittest.TestCase):
    def test_normalizes_feature_suffix(self):
        module = load_module()
        self.assertEqual(module.normalize_arch("gfx936:sramecc+:xnack-"), "gfx936")

    def test_accepts_eight_consistent_gfx906_devices(self):
        module = load_module()
        self.assertEqual(module.validate_arches(["gfx906"] * 8), "gfx906")

    def test_rejects_partial_eight_card_visibility(self):
        module = load_module()
        with self.assertRaisesRegex(ValueError, "Expected 8 devices, found 7"):
            module.validate_arches(["gfx936"] * 7)

    def test_rejects_mixed_architectures_with_observed_values(self):
        module = load_module()
        with self.assertRaisesRegex(ValueError, "gfx906.*gfx936|gfx936.*gfx906"):
            module.validate_arches(["gfx936"] * 7 + ["gfx906"])

    def test_rejects_blank_architecture(self):
        module = load_module()
        with self.assertRaisesRegex(ValueError, "Invalid Hygon architecture"):
            module.normalize_arch("")

    def test_rejects_unsupported_architecture(self):
        module = load_module()
        with self.assertRaisesRegex(ValueError, "Unsupported Hygon architecture: gfx942"):
            module.validate_arches(["gfx942"] * 8)


if __name__ == "__main__":
    unittest.main()

import importlib.metadata as metadata
from importlib.machinery import EXTENSION_SUFFIXES
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import runtime_probe as probe


class RuntimeProbeTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def dist(self, name, version, requires=()):
        path = self.root / f"{name}-{version}.dist-info"
        path.mkdir()
        text = f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
        text += "".join(f"Requires-Dist: {req}\n" for req in requires)
        (path / "METADATA").write_text(text, encoding="utf-8")
        return metadata.Distribution.at(path)

    def test_missing_transitive_dependency_triggers_repair_even_when_roots_exist(self):
        installed = {"httpx": self.dist("httpx", "0.28.1", ["httpcore==1.*"])}
        self.assertEqual(probe.dependency_problems(["httpx==0.28.1"], installed),
                         ["httpx 0.28.1: missing httpcore==1.*"])

    def test_wrong_installed_version_is_not_accepted_just_because_it_imports(self):
        installed = {"safetensors": self.dist("safetensors", "0.7.0")}
        self.assertEqual(probe.dependency_problems(["safetensors==0.8.0"], installed),
                         ["runtime requirements: requires safetensors==0.8.0; installed 0.7.0"])

    def test_unrelated_pandas_conflict_does_not_trigger_reinstall_loop(self):
        installed = {"janus": self.dist("janus", "2.0.0"),
                     "pandas": self.dist("pandas", "3.0.2", ["numpy>=1.26.0"])}
        self.assertEqual(probe.dependency_problems(["janus==2.0.0"], installed), [])

    def test_optional_dependencies_are_checked_only_when_extra_is_requested(self):
        installed = {"example": self.dist("example", "1.0", [
            'optional; extra == "video"', 'future; python_version < "2"'])}
        self.assertEqual(probe.dependency_problems(["example==1.0"], installed), [])
        self.assertEqual(probe.dependency_problems(["example[video]==1.0"], installed),
                         ['example 1.0: missing optional; extra == "video"'])

    def test_registration_inspection_does_not_import_package_or_native_extension(self):
        package = self.root / "python/infinilm"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text('raise RuntimeError("must not import in probe")\n')
        dist = self.dist("infinilm", "0.1.0")
        direct = self.root / "infinilm-0.1.0.dist-info/direct_url.json"
        direct.write_text(json.dumps({"url": self.root.as_uri(), "dir_info": {"editable": True}}))
        with patch.object(metadata, "distribution", return_value=dist), patch.object(sys, "path", [str(package.parent), *sys.path]):
            self.assertEqual(probe.check_registration("infinilm", self.root), 0)
            direct.write_text(json.dumps({"url": (self.root / "old").as_uri(), "dir_info": {"editable": True}}))
            self.assertEqual(probe.check_registration("infinilm", self.root), 10)

    def test_native_check_uses_already_loaded_canonical_module(self):
        package = self.root / "python/infinilm"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text('raise RuntimeError("must not import nested native module")\n')
        core = types.ModuleType("infinicore")
        public = types.ModuleType("infinilm")
        public.__file__ = str(package / "__init__.py")
        native = types.ModuleType("_infinilm")
        native.__file__ = str(package / "lib" / ("_infinilm" + EXTENSION_SUFFIXES[0]))
        with patch.dict(sys.modules, {"infinicore": core, "infinilm": public, "_infinilm": native}):
            self.assertEqual(probe.check_project_import("infinilm", self.root), 0)

    def test_constraints_keep_exact_vendor_torch_and_numpy_versions(self):
        versions = {"torch": "2.9.0+das.opt1.dtk2604", "numpy": "1.25.0"}
        def version(name):
            if name not in versions:
                raise metadata.PackageNotFoundError(name)
            return versions[name]
        path = self.root / "constraints.txt"
        with patch.object(metadata, "version", side_effect=version):
            probe.write_constraints(path)
        self.assertEqual(path.read_text(), "torch==2.9.0+das.opt1.dtk2604\nnumpy==1.25.0\n")


if __name__ == "__main__":
    unittest.main()

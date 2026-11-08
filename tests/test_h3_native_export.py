"""Reject unsafe native kernel ABIs before passing pointers to HIP."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


def fixture():
    fields = []
    for i in range(22):
        kind = 'global_buffer' if i in (0, 5, 10, 15, 20, 21) else 'by_value'
        fields.append(f'      - .offset: {8*i}\n        .size: 8\n        .value_kind: {kind}')
    assembly = ('.amdgpu_metadata\n---\namdhsa.kernels:\n  - .args:\n' + '\n'.join(fields) +
                '\n    .group_segment_fixed_size: 0\n    .kernarg_segment_size: 176\n'
                '    .max_flat_workgroup_size: 256\n    .name: gate\n'
                '    .wavefront_size: 64\namdhsa.target: amdgcn-amd-amdhsa--gfx906\n'
                '...\n.end_amdgpu_metadata\n')
    meta = dict(name='gate', target=dict(backend='hip', arch='gfx906', warp_size=64),
                num_warps=4, num_ctas=1, shared=0, global_scratch_size=0,
                profile_scratch_size=0, launch_cooperative_grid=False, enable_fp_fusion=False)
    return assembly, meta


class NativeExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = ROOT / 'scripts/h3-gated-native-v1/abi.py'
        if not path.exists():
            cls.api = None
            return
        spec = importlib.util.spec_from_file_location('h3_native_abi_test', path)
        cls.api = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.api)

    def api_required(self):
        self.assertIsNotNone(self.api, 'Native ABI validation is not implemented')
        return self.api

    def test_complete_dynamic_abi_includes_two_scratch_pointers(self):
        result = self.api_required().validate_abi(*fixture())
        self.assertEqual(result['argument_count'], 22)
        self.assertEqual(result['kernarg_bytes'], 176)

    def test_specialized_jit_kernel_cannot_be_launched_as_generic(self):
        api = self.api_required()
        source = next((ROOT / 'logs/2026-10-26-ninetoothed-jit-server/evidence/jit-gated-residual').rglob('*.amdgcn'))
        with self.assertRaises(ValueError):
            api.validate_abi(source.read_text(), fixture()[1])

    def test_rejects_missing_hidden_argument_and_wrong_scalar_width(self):
        api = self.api_required()
        asm, meta = fixture()
        for bad in (asm.replace('      - .offset: 168\n        .size: 8\n        .value_kind: global_buffer\n', ''),
                    asm.replace('.size: 8', '.size: 4', 1),
                    asm.replace('.offset: 16', '.offset: 12', 1)):
            with self.subTest():
                with self.assertRaises(ValueError):
                    api.validate_abi(bad, meta)

    def test_rejects_nonzero_scratch_fusion_and_wrong_target(self):
        api = self.api_required()
        asm, meta = fixture()
        for key, val in [('global_scratch_size', 64), ('profile_scratch_size', 64),
                         ('enable_fp_fusion', True), ('num_warps', 8),
                         ('target', dict(backend='hip', arch='gfx942', warp_size=64))]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                api.validate_abi(asm, dict(meta, **{key: val}))
        with self.assertRaises(ValueError):
            api.validate_abi(asm, {k: v for k, v in meta.items() if k != 'profile_scratch_size'})

    def test_rejects_wrong_symbol_target_and_duplicate_kernels(self):
        api = self.api_required()
        asm, meta = fixture()
        for bad in (asm.replace('.name: gate', '.name: wrong'),
                    asm.replace('--gfx906', '--gfx942'), asm + asm):
            with self.subTest(), self.assertRaises(ValueError):
                api.validate_abi(bad, meta)


if __name__ == '__main__':
    unittest.main()

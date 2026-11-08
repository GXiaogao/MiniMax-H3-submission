import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import shutil
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'artifacts/infinifabric-build-v1'
SOURCE=ROOT/'artifacts/infinifabric-migration-v3/verification-extract/InfiniLM/csrc'


class LoadSyncPatchTests(unittest.TestCase):
    def setUp(self):
        path=FOLDER/'prepare_minimax_h3_load_sync_trace.py'
        self.assertTrue(path.exists(),'load sync patcher missing')
        spec=importlib.util.spec_from_file_location('load_sync_patch',path)
        self.m=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.m)

    def post_base_rank(self):
        builder=(FOLDER/'build_infinilm_extension.sh').read_text()
        start=builder.index('    "${PYTHON}" -c \'\n',builder.index('patch_hygon_fl2va_no_cpu_runtime()'))+len('    "${PYTHON}" -c \'\n')
        payload=builder[start:builder.index('\n\' "${LOCAL_PROJECT_DIR}"',start)]
        files=['models/qwen3_vl/qwen3_vl_for_conditional_generation.cpp','models/qwen3_5/qwen3_5_vision.cpp',
            'engine/infer_engine.cpp','engine/rank_worker.cpp','layers/attention/backends/static_attn.cpp']
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for name in files:
                target=root/'csrc'/name;target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(SOURCE/name,target)
            with patch.object(sys,'argv',['base-patch',str(root)]):exec(compile(payload,'actual-base-builder-patch','exec'),{})
            return (root/'csrc/engine/rank_worker.cpp').read_bytes()

    def test_real_builder_patch_composition_preserves_no_cpu_allocation(self):
        data=self.post_base_rank()
        text=self.m.instrument_rank(data)
        self.assertIn('{0}, infinicore::DataType::I64, rank_info_.device',text)
        self.assertIn('weight_load_sync::stage',text)

    def test_rank_patch_preserves_strict_loading_and_final_sync(self):
        original=self.post_base_rank()
        text=self.m.instrument_rank(original)
        self.assertIn('model_->load_parameter(local_param_name, staged_param);',text)
        self.assertIn('auto staged_params = local_params;',text)
        self.assertIn('model_->load_parameters_no_sync(staged_params, local_params_strict);',text)
        self.assertIn('tensor = weight_load_sync::stage(tensor, rank_info_.device);',text)
        self.assertLess(text.index('model_->load_parameters_no_sync(staged_params'),
            text.index('infinicore::context::syncStream();',text.index('model_->load_parameters_no_sync(staged_params')))
        for data in (original+b'\n',text.encode(),(SOURCE/'engine/rank_worker.cpp').read_bytes()):
            with self.assertRaises(ValueError):self.m.instrument_rank(data)

    def test_staging_helper_has_both_barriers_and_retains_source_until_target_complete(self):
        text=(FOLDER/'h3_weight_load_sync.hpp').read_text()
        self.assertIn('Device::Type::HYGON',text)
        self.assertIn('source->device() == destination',text)
        self.assertLess(text.index('setDevice(source->device())'),text.index('source->contiguous()'))
        source_barrier=text.index('syncStream()',text.index('source->contiguous()'))
        transfer=text.index('ready->to(destination)')
        target_barrier=text.index('syncStream()',transfer)
        self.assertLess(source_barrier,transfer)
        self.assertLess(target_barrier,text.index('return staged;'))
        self.assertNotIn('from_torch',text)

    def test_all_sources_helpers_and_backups_are_recorded(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);h3=root/'h3/minimax_h3_transformer.cpp';h3.parent.mkdir()
            rank=root/'engine/rank_worker.cpp';rank.parent.mkdir()
            quant=root/'quant.cpp'
            h3.write_bytes((SOURCE/'models/minimax_h3/minimax_h3_transformer.cpp').read_bytes())
            rank.write_bytes(self.post_base_rank())
            quant.write_bytes((SOURCE/'layers/quantization/none_quantization.cpp').read_bytes())
            meta=self.m.patch_files(h3,quant,rank)
            self.assertEqual(meta['rank_patched_sha256'],self.m.sha(rank.read_bytes()))
            self.assertEqual(meta['helper_hashes']['h3_weight_load_sync.hpp'],self.m.sha((rank.parent/'h3_weight_load_sync.hpp').read_bytes()))
            self.assertTrue(rank.with_suffix('.cpp.before-load-sync').exists())
            self.assertEqual(meta['trace_version'],'1.0-isolated-h3-load-sync-projection-trace')
            with self.assertRaises((ValueError,FileExistsError)):self.m.patch_files(h3,quant,rank)

    def test_builder_isolated_and_records_rank_source_and_helper(self):
        builder=(FOLDER/'build_infinilm_extension.sh').read_bytes()
        text,meta=self.m.builder_text(builder)
        self.assertIn('/diagnostics/h3-load-sync/',text)
        self.assertIn('infinilm-load-sync.success',text)
        self.assertIn('--rank-source "${LOCAL_PROJECT_DIR}/csrc/engine/rank_worker.cpp"',text)
        self.assertIn('"${LOCAL_PROJECT_DIR}/csrc/engine/h3_weight_load_sync.hpp"',text)
        self.assertIn('"${LOCAL_PROJECT_DIR}/csrc/engine/rank_worker.cpp"',text)
        self.assertNotIn('"${STATE_DIR}/infinilm-extension.success"',text)
        with self.assertRaises(ValueError):self.m.builder_text(builder+b'\n')

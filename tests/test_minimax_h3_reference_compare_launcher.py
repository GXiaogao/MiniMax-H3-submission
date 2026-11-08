import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'artifacts/infinifabric-build-v1/run_minimax_h3_reference_compare.sh'
BASH=os.environ.get('TEST_BASH') or shutil.which('bash')

@unittest.skipUnless(BASH,'Bash required')
class ReferenceLauncherTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(SCRIPT.is_file(),'reference launcher missing')
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.env=os.environ.copy()
        for key in ('IFAB_AFS_HOME','WORK_HOME','LOG_DIR','PYTHON'):self.env.pop(key,None)

    def shell(self,text,*args):
        return subprocess.run([BASH,'-c','export PATH="/usr/bin:/bin:$PATH"\n'+text,'test',*map(str,args)],
            cwd=self.root,env=self.env,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=30)

    def test_overrides_rejected_before_writes(self):
        p=self.shell('bash "$1" --steps 2',SCRIPT.as_posix())
        self.assertEqual(p.returncode,2,p.stdout+p.stderr)
        self.assertEqual(list(self.root.iterdir()),[])

    def test_missing_afs(self):
        p=self.shell('bash "$1"',SCRIPT.as_posix())
        self.assertEqual(p.returncode,1);self.assertIn('IFAB_AFS_HOME',p.stderr)

    def test_failure_does_not_claim_comparison_and_keeps_exit(self):
        project=self.root/'project';project.mkdir();shutil.copyfile(SCRIPT,project/SCRIPT.name)
        for name in ('run_minimax_h3_reference_compare.py','run_minimax_h3_load_sync_generation.py',
                'diagnose_minimax_h3_full_forward_load_sync.py','diagnose_minimax_h3_load_sync.py',
                'diagnose_minimax_h3_prefix_parity.py','diagnose_minimax_h3_transformer_parity.py',
                'run_minimax_h3_fl2va_native_smoke.py','diagnose_minimax_h3_vae_roundtrip.py',
                'install_infinilm_private_runtime.py','prepare_minimax_h3_hygon_runtime.py','minimax_h3_compare.py'):(project/name).touch()
        work=self.root/'afs/minimax-h3';work.mkdir(parents=True)
        (work/'setup_infinifabric_env.sh').write_text('# fixture\n')
        stub=project/'python.sh'
        stub.write_text('#!/usr/bin/env bash\nif [[ "$1" == "-c" ]]; then echo /torch/lib; exit 0; fi\necho reference-failed >&2\nexit 36\n',newline='\n')
        self.env.update(IFAB_AFS_HOME=work.parent.as_posix(),PYTHON=stub.as_posix())
        p=self.shell('if command -v cygpath >/dev/null; then export IFAB_AFS_HOME="$(cygpath -u "$IFAB_AFS_HOME")"; fi\nbash "$1"',(project/SCRIPT.name).as_posix())
        self.assertEqual(p.returncode,36,p.stdout+p.stderr)
        logs=list((work/'logs').rglob('run.log'));self.assertEqual(len(logs),1)
        self.assertIn('reference-failed',logs[0].read_text())
        self.assertIn('exit_code=36',(logs[0].parent/'summary.txt').read_text())
        self.assertNotIn('QUALITY_COMPARISON_COMPLETE',p.stdout)

if __name__=='__main__':unittest.main()

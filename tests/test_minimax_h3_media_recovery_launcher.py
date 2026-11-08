import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'artifacts/infinifabric-build-v1/recover_minimax_h3_generation_media.sh'
BASH=os.environ.get('TEST_BASH') or shutil.which('bash')

@unittest.skipUnless(BASH,'Bash required')
class MediaRecoveryLauncherTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(SCRIPT.is_file(),'recovery launcher missing')
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.env=os.environ.copy()
        for key in ('IFAB_AFS_HOME','WORK_HOME','LOG_DIR','PYTHON'):self.env.pop(key,None)

    def run_shell(self,text,*args):
        return subprocess.run([BASH,'-c','export PATH="/usr/bin:/bin:$PATH"\n'+text,'test',*map(str,args)],
            cwd=self.root,env=self.env,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=30)

    def test_no_args_override_or_missing_afs(self):
        p=self.run_shell('bash "$1" --steps 49',SCRIPT.as_posix())
        self.assertEqual(p.returncode,2,p.stdout+p.stderr)
        self.assertEqual(list(self.root.iterdir()),[])
        p=self.run_shell('bash "$1"',SCRIPT.as_posix())
        self.assertEqual(p.returncode,1);self.assertIn('IFAB_AFS_HOME',p.stderr)

    def test_failed_recovery_is_retained_and_does_not_claim_media_or_generation(self):
        project=self.root/'project';project.mkdir();shutil.copyfile(SCRIPT,project/SCRIPT.name)
        for name in ('recover_minimax_h3_generation_media.py','run_minimax_h3_load_sync_generation.py',
                'diagnose_minimax_h3_full_forward_load_sync.py','diagnose_minimax_h3_load_sync.py',
                'diagnose_minimax_h3_prefix_parity.py','diagnose_minimax_h3_transformer_parity.py',
                'run_minimax_h3_fl2va_native_smoke.py','install_infinilm_private_runtime.py'):(project/name).touch()
        work=self.root/'afs/minimax-h3';work.mkdir(parents=True)
        (work/'setup_infinifabric_env.sh').write_text('# fixture\n')
        stub=project/'python.sh';stub.write_text('#!/usr/bin/env bash\necho media-inspection-failed >&2\nexit 38\n',newline='\n')
        self.env.update(IFAB_AFS_HOME=work.parent.as_posix(),PYTHON=stub.as_posix())
        p=self.run_shell('if command -v cygpath >/dev/null; then export IFAB_AFS_HOME="$(cygpath -u "$IFAB_AFS_HOME")"; fi\nbash "$1"',(project/SCRIPT.name).as_posix())
        self.assertEqual(p.returncode,38,p.stdout+p.stderr)
        logs=list((work/'logs').rglob('run.log'));self.assertEqual(len(logs),1)
        self.assertIn('media-inspection-failed',logs[0].read_text())
        self.assertIn('exit_code=38',(logs[0].parent/'summary.txt').read_text())
        self.assertNotIn('EXISTING_MEDIA_VALIDATED',p.stdout)

if __name__=='__main__':unittest.main()

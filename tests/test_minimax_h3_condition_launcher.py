import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'artifacts/infinifabric-build-v1/build_minimax_h3_condition_trace.sh'
BASH=os.environ.get('TEST_BASH') or shutil.which('bash')

@unittest.skipUnless(BASH,'Bash required')
class ConditionBuildLauncherTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(SCRIPT.is_file(),'parity launcher missing')
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.env=os.environ.copy()
        for k in ('IFAB_AFS_HOME','WORK_HOME','LOG_DIR','PYTHON'): self.env.pop(k,None)

    def run_shell(self, script, *args):
        return subprocess.run([BASH,'-c','export PATH="/usr/bin:/bin:$PATH"\n'+script,'test',*map(str,args)],
                              cwd=self.root,env=self.env,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=30)

    def test_overrides_rejected_before_any_writes(self):
        result=self.run_shell('bash "$1" --steps 1',SCRIPT.as_posix())
        self.assertEqual(result.returncode,2,result.stdout+result.stderr)
        self.assertEqual(list(self.root.iterdir()),[])

    def test_missing_afs_is_explicit(self):
        result=self.run_shell('bash "$1"',SCRIPT.as_posix())
        self.assertEqual(result.returncode,1)
        self.assertIn('IFAB_AFS_HOME',result.stderr)

    def test_failed_diagnostic_is_retained_and_never_prints_complete(self):
        project=self.root/'project';project.mkdir()
        shutil.copyfile(SCRIPT,project/SCRIPT.name)
        for n in ('prepare_minimax_h3_condition_trace.py','prepare_minimax_h3_boundary_trace.py','h3_boundary_trace.hpp','build_infinilm_extension.sh'): (project/n).touch()
        work=self.root/'afs/minimax-h3';work.mkdir(parents=True)
        (work/'setup_infinifabric_env.sh').write_text('# fixture\n')
        stub=project/'python.sh'
        stub.write_text('#!/usr/bin/env bash\nif [[ "$1" == "-c" ]]; then echo /torch/lib; exit 0; fi\necho reference-load-failed >&2\nexit 39\n',newline='\n')
        self.env.update(IFAB_AFS_HOME=work.parent.as_posix(),PYTHON=stub.as_posix())
        result=self.run_shell('if command -v cygpath >/dev/null; then export IFAB_AFS_HOME="$(cygpath -u "$IFAB_AFS_HOME")"; fi\nbash "$1"',(project/SCRIPT.name).as_posix())
        self.assertEqual(result.returncode,39,result.stdout+result.stderr)
        logs=list((work/'logs').rglob('run.log'))
        self.assertEqual(len(logs),1)
        self.assertIn('reference-load-failed',logs[0].read_text())
        self.assertIn('exit_code=39',(logs[0].parent/'summary.txt').read_text())
        self.assertIn('last_stage=prepare-isolated-builder',(logs[0].parent/'summary.txt').read_text())
        self.assertNotIn('ISOLATED_CONDITION_BUILD COMPLETE',result.stdout)

    def test_actual_build_child_failure_preserves_exit_and_stage(self):
        project=self.root/'project';project.mkdir()
        shutil.copyfile(SCRIPT,project/SCRIPT.name)
        for n in ('prepare_minimax_h3_condition_trace.py','prepare_minimax_h3_boundary_trace.py','h3_boundary_trace.hpp','build_infinilm_extension.sh'):(project/n).touch()
        work=self.root/'afs/minimax-h3';work.mkdir(parents=True)
        (work/'setup_infinifabric_env.sh').write_text('# fixture\n')
        stub=project/'python.sh'
        stub.write_text('''#!/usr/bin/env bash
while [[ "$#" -gt 0 ]]; do
    if [[ "$1" == "--destination" ]]; then
        shift
        printf '#!/usr/bin/env bash\\necho boundary-build-failed >&2\\nexit 39\\n' > "$1"
        exit 0
    fi
    shift
done
exit 2
''',newline='\n')
        self.env.update(IFAB_AFS_HOME=work.parent.as_posix(),PYTHON=stub.as_posix())
        result=self.run_shell('if command -v cygpath >/dev/null; then export IFAB_AFS_HOME="$(cygpath -u "$IFAB_AFS_HOME")"; fi\nbash "$1"',(project/SCRIPT.name).as_posix())
        self.assertEqual(result.returncode,39,result.stdout+result.stderr)
        log=next((work/'logs').rglob('run.log'))
        self.assertIn('boundary-build-failed',log.read_text())
        self.assertIn('last_stage=build',(log.parent/'summary.txt').read_text())
        self.assertNotIn('ISOLATED_CONDITION_BUILD COMPLETE',result.stdout)

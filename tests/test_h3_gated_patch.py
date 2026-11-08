"""Atomic source application must preserve unrelated edits and reject unknown input."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import hashlib

ROOT = Path(__file__).resolve().parents[1]


class GatedPatchTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location('gated_patch_test', ROOT/'scripts/h3-gated-integration-v1/apply_patch.py')
        self.assertTrue(Path(spec.origin).is_file(), 'Transactional source installer not implemented')
        self.api = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.api)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.repos = {n:self.root/n for n in ('InfiniCore','InfiniLM')}
        self.package = self.root/'package'
        self.package.mkdir()
        self.manifest = {'files':[],'prerequisites':[]}
        for name,repo in self.repos.items():
            repo.mkdir()
            (repo/'existing.cc').write_bytes(b'baseline\r\n')
            (repo/'unrelated.txt').write_bytes(b'user edit')
            for path,before,data in [('existing.cc',b'baseline\n',b'updated\n'),('new.cc',None,b'new file\n')]:
                payload=self.package/name/path;payload.parent.mkdir(exist_ok=True);payload.write_bytes(data)
                self.manifest['files'].append({'repo':name,'path':path,'before':hashlib.sha256(before).hexdigest() if before else None,'after':hashlib.sha256(data).hexdigest()})

    def test_applies_both_repositories_and_is_idempotent(self):
        self.api.install(self.repos,self.package,self.manifest,self.root/'backup')
        for repo in self.repos.values():
            self.assertEqual((repo/'existing.cc').read_bytes(),b'updated\n')
            self.assertEqual((repo/'new.cc').read_bytes(),b'new file\n')
            self.assertEqual((repo/'unrelated.txt').read_bytes(),b'user edit')
        self.assertEqual(self.api.install(self.repos,self.package,self.manifest,self.root/'backup2'),'already-applied')

    def test_unknown_second_repo_edit_prevents_any_write(self):
        (self.repos['InfiniLM']/'existing.cc').write_bytes(b'unknown edit')
        with self.assertRaises(RuntimeError):
            self.api.install(self.repos,self.package,self.manifest,self.root/'backup')
        self.assertEqual((self.repos['InfiniCore']/'existing.cc').read_bytes(),b'baseline\r\n')
        self.assertFalse((self.repos['InfiniCore']/'new.cc').exists())

    def test_corrupt_payload_and_path_escape_are_rejected(self):
        for corrupt in ('hash','path'):
            with self.subTest(corrupt=corrupt):
                m=json.loads(json.dumps(self.manifest))
                m['files'][0]['after' if corrupt=='hash' else 'path']='bad' if corrupt=='hash' else '../outside'
                with self.assertRaises((RuntimeError,ValueError)):
                    self.api.install(self.repos,self.package,m,self.root/('backup_'+corrupt))

    def test_write_failure_restores_original_bytes(self):
        actual=self.api.replace_file
        def fail_on_lm(path,data):
            if path.parent==self.repos['InfiniLM'] and data==b'updated\n':
                raise OSError('simulated disk failure')
            actual(path,data)
        self.api.replace_file=fail_on_lm
        with self.assertRaises(OSError):
            self.api.install(self.repos,self.package,self.manifest,self.root/'backup')
        for repo in self.repos.values():
            self.assertEqual((repo/'existing.cc').read_bytes(),b'baseline\r\n')
            self.assertFalse((repo/'new.cc').exists())


if __name__=='__main__': unittest.main()

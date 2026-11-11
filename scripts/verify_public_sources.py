"""Verify report-scoped public Git checkouts and selected source evidence; no GPU work."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def verify(checkouts):
    rows=json.loads((ROOT/'configs/public-repositories.json').read_text(encoding='utf-8'))
    verified=[]
    for row in rows:
        repo=checkouts/row['component']
        head=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
        if head!=row['commit']:raise ValueError('wrong public commit: '+row['component'])
        dirty=subprocess.check_output(['git','-C',str(repo),'status','--porcelain','--untracked-files=no'],text=True)
        if dirty:raise ValueError('tracked source modified: '+row['component'])
        record=json.loads((repo/'competition/source-provenance.json').read_text(encoding='utf-8'))
        if record['upstream_commit']!=row['upstream_commit']:raise ValueError('wrong upstream identity')
        for relative,expected in row['source_hashes'].items():
            actual=hashlib.sha256((repo/relative).read_bytes()).hexdigest()
            if actual!=expected:raise ValueError('source SHA256 differs: '+relative)
        verified.append(dict(component=row['component'],commit=head,checked_files=len(row['source_hashes'])))
    return dict(result='SOURCE_IDENTITY_VERIFIED_NOT_DEVICE_ACCEPTANCE',repositories=verified)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkouts',type=Path,required=True)
    print(json.dumps(verify(p.parse_args().checkouts),indent=2))

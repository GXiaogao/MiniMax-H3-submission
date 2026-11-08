"""Create an LF-only offline benchmark package without changing validated artifacts."""
from pathlib import Path
import gzip
import hashlib
import io
import tarfile

ROOT=Path(__file__).resolve().parents[1]
PACKAGE=ROOT/'scripts/h3-gated-benchmark-v1'


def main():
    # Immutable verification helpers from the already accepted v3 package.
    for name in ('rounding_reference.py','check_sources.py','patch_manifest.json'):
        (PACKAGE/name).write_bytes((ROOT/'scripts/h3-gated-rounding-v3'/name).read_bytes().replace(b'\r\n',b'\n'))
    files=sorted(p for p in PACKAGE.iterdir() if p.is_file() and p.name!='SHA256SUMS')
    for path in files:path.write_bytes(path.read_bytes().replace(b'\r\n',b'\n'))
    sums=PACKAGE/'SHA256SUMS'
    sums.write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.name+'\n' for p in files),encoding='ascii',newline='\n')
    files.append(sums)
    target=ROOT/'artifacts/h3-gated-benchmark-v1.tar.gz'
    with target.open('wb') as raw,gzip.GzipFile(fileobj=raw,mode='wb',filename='',mtime=0) as gz,tarfile.open(fileobj=gz,mode='w') as archive:
        for path in files:
            data=path.read_bytes();entry=tarfile.TarInfo(PACKAGE.name+'/'+path.name)
            entry.size=len(data);entry.mode=0o755 if path.suffix=='.sh' else 0o644
            archive.addfile(entry,io.BytesIO(data))
    digest=hashlib.sha256(target.read_bytes()).hexdigest()
    target.with_name(target.name+'.sha256').write_text(digest+'  '+target.name+'\n',encoding='ascii',newline='\n')
    print(target);print('SHA256:',digest,'files:',len(files))


if __name__=='__main__':main()

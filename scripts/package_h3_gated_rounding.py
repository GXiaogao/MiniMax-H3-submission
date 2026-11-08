"""Package v3 without changing the archived v1/v2 source packages or artifacts."""
from pathlib import Path
import gzip
import hashlib
import io
import tarfile

ROOT=Path(__file__).resolve().parents[1]
PACKAGE=ROOT/'scripts/h3-gated-rounding-v3'


def main():
    files=sorted(p for p in PACKAGE.iterdir() if p.is_file() and p.name!='SHA256SUMS')
    for path in files:path.write_bytes(path.read_bytes().replace(b'\r\n',b'\n'))
    checksum=PACKAGE/'SHA256SUMS'
    checksum.write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.name+'\n' for p in files),encoding='ascii',newline='\n')
    files.append(checksum)
    target=ROOT/'artifacts/h3-gated-rounding-v3.tar.gz'
    with target.open('wb') as raw,gzip.GzipFile(fileobj=raw,mode='wb',filename='',mtime=0) as gz,tarfile.open(fileobj=gz,mode='w') as tar:
        for path in files:
            data=path.read_bytes();entry=tarfile.TarInfo(PACKAGE.name+'/'+path.name)
            entry.size=len(data);entry.mode=0o755 if path.suffix=='.sh' else 0o644
            tar.addfile(entry,io.BytesIO(data))
    print(target)
    print('SHA256:',hashlib.sha256(target.read_bytes()).hexdigest(),'members:',len(files))


if __name__=='__main__':main()

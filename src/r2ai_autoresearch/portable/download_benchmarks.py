"""Download pinned benchmark files; preserve dataset access controls and hashes."""
import argparse
import hashlib
import os
from pathlib import Path
import urllib.request
from .benchmark_sources import SOURCES


def download(output):
    for relative,source in SOURCES.items():
        path=Path(output)/relative
        if path.exists():
            if hashlib.sha256(path.read_bytes()).hexdigest()!=source['sha256']:raise ValueError('cached dataset hash mismatch: '+relative)
            continue
        headers={}
        if source['url'].startswith('https://huggingface.co/') and os.getenv('HF_TOKEN'):
            headers['Authorization']='Bearer '+os.environ['HF_TOKEN']
        try:
            with urllib.request.urlopen(urllib.request.Request(source['url'],headers=headers),timeout=120) as f: data=f.read()
        except Exception:
            raise RuntimeError('Could not download '+relative+'. For gated WildJailbreak, accept the dataset terms and set HF_TOKEN, or provide your existing authorized evaluation-root. No token is logged.') from None
        if hashlib.sha256(data).hexdigest()!=source['sha256']:raise ValueError('download hash mismatch: '+relative)
        path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_bytes(data);tmp.replace(path)
        print('Verified '+relative,flush=True)
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    download(p.parse_args().output)

if __name__=='__main__':main()

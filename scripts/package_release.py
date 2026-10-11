"""Copy the public source allowlist; omit weights, runs, caches and credentials."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[1];dest=Path(a.output).resolve()
    if dest==root or root in dest.parents:raise ValueError('Export outside the source checkout')
    if dest.exists() and any(dest.iterdir()):
        if not (dest/'release_manifest.json').is_file():raise ValueError('Refusing to overwrite an unrelated directory')
    manifest={}
    for line in (root/'release-files.txt').read_text().splitlines():
        if not line or line.startswith('#'):continue
        relative=Path(line)
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Invalid source path: '+line)
        source=root/relative
        if not source.is_file() or source.is_symlink():raise ValueError('Missing file or symlink in release: '+line)
        target=dest/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        manifest[line]=hashlib.sha256(target.read_bytes()).hexdigest()
    # Fail rather than delete unexpected files from an existing export.
    actual={str(f.relative_to(dest)) for f in dest.rglob('*') if f.is_file() and not any(part in ('__pycache__','build','dist','.runtime') or part.endswith('.egg-info') or part.startswith('.venv') for part in f.relative_to(dest).parts) and f.name!='release_manifest.json'}
    extras=actual-set(manifest)
    if extras:raise ValueError('Export has unlisted files: '+', '.join(sorted(extras)))
    (dest/'release_manifest.json').write_text(json.dumps({'schema':'r2ai.release.v1','files':manifest},indent=2)+'\n')
    print(json.dumps({'output':str(dest),'files':len(manifest)}))

if __name__=='__main__':main()

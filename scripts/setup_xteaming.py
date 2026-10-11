"""Fetch pinned upstream X-Teaming protocol prompts without republishing them."""
import argparse,hashlib,json,urllib.request
from pathlib import Path
REV='db964e6630b6a69b78e093705c89fd34865b6c27'
FILES={'attacker_agent_prompts.yaml':'5172d1ef83b831fb3f421623c34a2b509cb10110eefaa9673063986af5ecd2b9',
       'plan_generation_prompts.yaml':'806a98a95aacc2bcbb3304e8665a61391a80ba4cc2eb3c85c3486751190994df'}
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default=str(Path(__file__).resolve().parents[1]/'.runtime/xteaming-prompts'));a=p.parse_args()
    root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
    for name,expected in FILES.items():
        dest=root/name
        if dest.is_file() and hashlib.sha256(dest.read_bytes()).hexdigest()==expected:continue
        url=f'https://raw.githubusercontent.com/salman-lui/x-teaming/{REV}/config/prompts/{name}'
        data=urllib.request.urlopen(url,timeout=60).read()
        if hashlib.sha256(data).hexdigest()!=expected:raise ValueError('Upstream prompt hash mismatch: '+name)
        temp=dest.with_suffix('.tmp');temp.write_bytes(data);temp.replace(dest)
    (root/'provenance.json').write_text(json.dumps({'upstream':'https://github.com/salman-lui/x-teaming','revision':REV,'sha256':FILES},indent=2)+'\n')
    print(root)
if __name__=='__main__':main()

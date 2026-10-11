"""Export complete trial results as CSV and Markdown without exposing prompts."""
import argparse,csv,json
from pathlib import Path

FIELDS=['trial','phase','source','method','samples','steps','learning_rate','eligible','safety_gain','primary_asr','secondary_asr','benign_compliance','instruction_following','issues','parent','checkpoint']

def report(run,output=None):
    run=Path(run).resolve();dest=Path(output).resolve() if output else run/'report';dest.mkdir(parents=True,exist_ok=True)
    s=json.loads((run/'summary.json').read_text());rows=[]
    for t in s['trials']:
        r=t['recipe'];m=t.get('metrics') or {};g=t['gate']
        rows.append(dict(zip(FIELDS,[t['id'],t['phase'],'+'.join(r.get('alpha',{r['source']:r['sample_count']})),r['method'],sum(r.get('alpha',{r['source']:r['sample_count']}).values()),r.get('steps'),r.get('learning_rate'),g['eligible'],g['safety_gain'] if m else None,m.get('primary_asr'),m.get('secondary_asr'),m.get('benign_response'),m.get('instruction_constraint'),';'.join(g['issues']),t['parent'],t['checkpoint']])))
    with (dest/'trials.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerows(rows)
    lines=['# Research run results','',f"Retained checkpoint: `{s['retained_model']}`",'', 'Ratios are in [0,1]; multiply by 100 for percentages. Safety gain is an absolute ASR reduction, not a relative percentage.','', '| Round | Pilots/probes | Full trials | Eligible | Selected | Stop reason |','|---|---:|---:|---:|---|---|']
    for h in s['history']:lines.append(f"| {h['round']} | {len(h['pilots'])} | {len(h['full_trials'])} | {len(h['eligible'])} | {h['selected'] or 'none'} | {h['stop_reason']} |")
    lines+=['','See `trials.csv` for every completed or failed-signal trial. Failed or missing metrics are blank, never zero.','']
    (dest/'README.md').write_text('\n'.join(lines));return {'report':str(dest),'trials':len(rows)}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--output');a=p.parse_args();print(json.dumps(report(a.run,a.output),indent=2))
if __name__=='__main__':main()

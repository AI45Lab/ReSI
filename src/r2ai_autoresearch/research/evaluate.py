"""Evaluate a saved checkpoint independently of recipe search; export metric counts."""
import argparse,csv,json,os,fcntl
from pathlib import Path
from .config import load,bind_services
from .backend import Backend
from ..portable.contracts import save,digest
from ..portable.service import Client
from ..executor import sha256_path

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    group=p.add_mutually_exclusive_group(required=True);group.add_argument('--model');group.add_argument('--run')
    p.add_argument('--data');p.add_argument('--reference-model');p.add_argument('--output',required=True)
    a=p.parse_args();c=load(a.config);bind_services(c)
    model=a.model or json.loads((Path(a.run)/'summary.json').read_text())['retained_model'];model=str(Path(model).resolve())
    from .model_adapter import validate_checkpoint
    validate_checkpoint(model)
    if a.reference_model:validate_checkpoint(a.reference_model)
    data_path=Path(a.data or c['data']['path']);data=json.loads(data_path.read_text())
    c['profile']='smoke' if data['role']=='authored_engineering_examples' else 'standard'
    os.environ['CUDA_VISIBLE_DEVICES']=c['runtime']['gpu_ids']
    dest=Path(a.output).resolve();dest.mkdir(parents=True,exist_ok=True)
    lock=(dest/'evaluation.lock').open('a');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    src=Path(__file__).resolve().parents[1]
    identity={'config':c,'model':sha256_path(Path(model)),'data':digest(data),
              'reference':sha256_path(Path(a.reference_model)) if a.reference_model else None,
              'code':{str(f.relative_to(src)):sha256_path(f) for f in src.rglob('*.py')},
              'judge':{'model':Client.endpoint('JUDGE')[1],'endpoint_digest':digest(Client.endpoint('JUDGE')[0])}}
    frozen=dest/'evaluation_identity.json'
    if frozen.exists() and json.loads(frozen.read_text())!=identity:raise ValueError('Evaluation identity changed; choose a new output directory')
    save(frozen,identity);b=Backend(c,data,model,dest)
    jobs=[('target',model)]+([('reference',str(Path(a.reference_model).resolve()))] if a.reference_model else [])
    records={}
    for name,checkpoint in jobs:
        b.evaluate(name,checkpoint);records[name]=json.loads((dest/name/'evaluation.json').read_text())
    save(dest/'evaluation.json',{'model':model,'dataset_usage':data.get('benchmarks',{}).get('usage','engineering_demo'),'results':records,'selection_performed':False})
    with (dest/'metrics.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['model','metric','value','percent','numerator','denominator'])
        for name,r in records.items():
            for metric,value in r['metrics'].items():
                count=r['counts'].get(metric,{});w.writerow([name,metric,value,100*value,count.get('numerator'),count.get('denominator')])
    print(json.dumps({'output':str(dest),'metrics':{n:r['metrics'] for n,r in records.items()},'selection_performed':False},indent=2))
if __name__=='__main__':main()

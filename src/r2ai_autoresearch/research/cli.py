"""Run the manuscript-aligned controller from one JSON configuration."""
import argparse
import fcntl
import json
import os
from pathlib import Path
from .config import load,bind_services
from ..portable.contracts import save,digest
from ..executor import sha256_path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True)
    p.add_argument('--check',action='store_true',help='Validate config, data and endpoint declarations; no downloads, API or GPU')
    a=p.parse_args();c=load(a.config);bind_services(c)
    errors=[];data=None
    if not c['data']['path'] or not Path(c['data']['path']).is_file():errors.append('Set data.path to an existing training/validation manifest')
    else:
        data=json.loads(Path(c['data']['path']).read_text())
        if c['profile']=='standard' and data.get('safety_validation_origin')!='training_pool_holdout':
            errors.append('Standard profile requires safety_validation_origin=training_pool_holdout; do not reuse final-test safety data')
        if c['profile']=='standard':
            safety=[r for r in data['rows'] if r.get('kind') in ('harmful','benign') and (r.get('split')=='train' or r.get('benchmark')=='wildjailbreak')]
            if any(not r.get('seed_id') for r in safety):errors.append('Standard safety rows require seed_id lineage')
            train_ids={r['seed_id'] for r in safety if r.get('split')=='train' and r.get('seed_id')}
            heldout_ids={r['seed_id'] for r in safety if r.get('split')!='train' and r.get('seed_id')}
            if train_ids & heldout_ids:errors.append('Training and validation seed lineages overlap')
    from ..portable.service import Client
    resolved_services={}
    for role in ['AGENT','CRITIC','JUDGE',*('COLLECT_'+a.upper() for a in c['data']['providers'])]:
        try:
            url,service_model,_=Client.endpoint(role)
            resolved_services[role.lower()]={'base_url':url,'model':service_model}
        except RuntimeError:errors.append('Configure services.'+role.lower()+'.base_url/model (or corresponding environment declarations)')
    if data:
        # Validate complete data schema without allocating an output directory.
        from ..portable.contracts import validate_data
        validate_data(data,{'role':'engineering_demo' if c['profile']=='smoke' else 'benchmark_development','sources':c['data']['sources'],'examples_per_source':c['search']['n0'],'search':{'mode':'three_stage'}})
    for source,provider in c['data']['providers'].items():
        if provider['kind']=='jailbreak_skill':
            for skill in provider['skills']:
                if not (Path(provider['repository'])/'skills'/skill/'scripts/run.py').is_file():errors.append('Install JailbreakSkill package: '+skill)
    if a.check:
        print(json.dumps({'valid':not errors,'profile':c['profile'],'search':c['search'],'missing':errors,'gpu_validated':False},indent=2))
        if errors:raise SystemExit(2)
        return
    if errors:raise ValueError('; '.join(errors))
    os.environ['CUDA_VISIBLE_DEVICES']=c['runtime']['gpu_ids']
    model=c['model']['path']
    if not model:
        if not c['model']['revision']:raise ValueError('Pinned model revision required for downloading')
        from huggingface_hub import snapshot_download
        model=snapshot_download(c['model']['id'],revision=c['model']['revision'],cache_dir=c['runtime']['cache'],allow_patterns=['*.json','*.safetensors','*.txt','*.jinja','*.model','*.py'])
    model=Path(model)
    if not (model/'config.json').is_file() or not (list(model.glob('*.safetensors')) or (model/'r2ai_checkpoint.json').is_file()):raise ValueError('A complete local model checkpoint is required')
    from .model_adapter import describe, validate_checkpoint
    validate_checkpoint(model)
    capabilities=describe(model,c['model'])
    root=Path(c['runtime']['output']);root.mkdir(parents=True,exist_ok=True)
    source=Path(__file__).resolve().parents[1]
    generator_code={}
    for name,provider in c['data']['providers'].items():
        if provider['kind']=='jailbreak_skill':
            repo=Path(provider['repository'])
            folders=[repo/'core',*(repo/'skills'/skill for skill in provider['skills'])]
            generator_code[name]={str(f.relative_to(repo)):sha256_path(f) for folder in folders for f in folder.rglob('*') if f.is_file() and f.suffix in ('.py','.md','.json','.yaml')}
    package=source.parents[1]
    execution_files=[f for folder in (package/'scripts',package/'patches') for f in folder.rglob('*') if f.is_file() and f.suffix in ('.py','.sh','.patch')]
    execution_files.extend(package.glob('requirements*.txt'))
    execution_identity={str(f.relative_to(package)):sha256_path(f) for f in execution_files}
    fingerprint={'execution_files':execution_identity,'generator_code':generator_code,'backend_capabilities':capabilities,'config':c,'services':resolved_services,'data':digest(data),'model':sha256_path(model),'code':{str(f.relative_to(source)):sha256_path(f) for f in source.rglob('*.py')}}
    with (root/'controller.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        frozen=root/'run_identity.json'
        if frozen.exists() and json.loads(frozen.read_text())!=fingerprint:raise ValueError('Changed config/data/model/code: choose a new output directory')
        save(frozen,fingerprint);save(root/'resolved_config.json',c)
        if (root/'summary.json').exists():
            result=json.loads((root/'summary.json').read_text())
            if sha256_path(Path(result['retained_model']))!=result['retained_model_sha256']:raise ValueError('Retained checkpoint has changed')
        else:
            from .backend import Backend
            from .controller import Controller
            backend=Backend(c,data,model,root)
            try:result=Controller(c,backend).run()
            except Exception as e:
                save(root/'status.json',{'status':'failed','error_type':type(e).__name__});raise
            result['retained_model_sha256']=sha256_path(Path(result['retained_model']))
            save(root/'summary.json',result);save(root/'status.json',{'status':'completed'})
        print(json.dumps({'retained_model':result['retained_model'],'history':result['history']},indent=2))

if __name__=='__main__':main()

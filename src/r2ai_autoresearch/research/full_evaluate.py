"""Run the migrated final-benchmark suite from the same config.json."""
import argparse
import csv
import json
import os
from pathlib import Path
from .config import load, bind_services
from ..portable.contracts import save, digest
from ..portable.service import Client


def default_settings():
    return {'benchmarks':['wildjailbreak','harmbench','h_cot','xteaming','xstest','gpqa_diamond','mmlu_pro','ifeval'],
            'target':{'base_url':None,'model':None,'api_key_env':'R2AI_TARGET_API_KEY'},
            'output':'runs/final-evaluation','workers':4,
            'requests':{'target_timeout_seconds':3600,'max_retries':2,'transient_retries':3},
            'xteaming':{'behavior_path':None,'behavior_count':159,'planner_sets':5,'plans_per_set':10,
                        'plans_to_execute':10,'max_turns':7,'target_reasoning_policy':'answer_only','planner':{'base_url':None,'model':'DeepSeek-R1','api_key_env':None,'timeout_seconds':None},'attacker':{'base_url':None,'model':'Qwen2.5-32B-Instruct','api_key_env':None,'timeout_seconds':None},'max_textgrad_iterations':4}}


def attack_roles(settings,agent,judge,target,generation,extra,system_prompt):
    def resolve(name):
        declared=settings[name]
        return {**agent,**{k:v for k,v in declared.items() if v is not None}}
    attacker=resolve('attacker');planner=resolve('planner')
    return {'planner':planner,'attacker':attacker,'optimizer':attacker,
            'verifier':{**judge,'temperature':0.,'max_tokens':1024},
            'target':{**target,'temperature':generation.get('temperature'),'top_p':generation.get('top_p'),
                      'max_tokens':generation['max_tokens'],'generation_extra_body':extra,'system_prompt':system_prompt}}


def prepare(c, config_path):
    from ..formal_eval.config import Endpoint
    settings=c['final_evaluation'];root=Path(config_path).resolve().parent
    policy=settings['requests']
    retries={k:policy[k] for k in ('max_retries','transient_retries')}
    target={**settings['target'],**retries,'timeout_seconds':policy['target_timeout_seconds']};target.setdefault('api_key_env','R2AI_TARGET_API_KEY')
    if not target.get('base_url') or not target.get('model'):raise ValueError('Set final_evaluation.target.base_url and model')
    Endpoint.from_mapping(target,source_path=config_path)
    url,model,key=Client.endpoint('JUDGE');os.environ['R2AI_FULL_EVAL_JUDGE_KEY']=key
    judge={'base_url':url,'model':model,'api_key_env':'R2AI_FULL_EVAL_JUDGE_KEY',
           'timeout_seconds':c['services']['judge']['timeout_seconds'],**retries}
    generation=dict(c['evaluation']['generation'])
    if c['model']['path']:
        model_path=Path(c['model']['path']);model_config=model_path/'config.json'
        if model_config.is_file() and json.loads(model_config.read_text()).get('model_type')=='mistral3':
            native=(model_path/'SYSTEM_PROMPT.txt').read_text().strip()
            if c['evaluation']['system_prompt'] not in ('',native):raise ValueError('Ministral evaluation requires its native system prompt')
            c['evaluation']['system_prompt']=native
        path=model_path/'generation_config.json'
        if path.is_file():
            defaults=json.loads(path.read_text())
            generation={k:(defaults.get(k) if v is None else v) for k,v in generation.items()}
    # Null temperature/top-p omit overrides and preserve serving defaults.
    if generation.get('do_sample') is False:generation['temperature']=0.
    extra={k:generation[k] for k in ('top_k',) if generation.get(k) is not None}
    extra['chat_template_kwargs']={'enable_thinking':c['model']['enable_thinking']}
    dest=(root/settings['output']).resolve()
    return target,judge,generation,extra,dest


def numeric_metrics(metrics, prefix=''):
    for key,value in metrics.items():
        name=prefix+str(key)
        if isinstance(value,dict):yield from numeric_metrics(value,name+'.')
        elif type(value) in (int,float):yield name,value


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    g=p.add_mutually_exclusive_group();g.add_argument('--run',nargs='?',const='@configured',help='Completed search directory; omitted value uses runtime.output');g.add_argument('--checkpoint')
    p.add_argument('--external-target',action='store_true',help='Use an explicitly managed endpoint instead of launching the retained model')
    p.add_argument('--export-dir');p.add_argument('--output');p.add_argument('--startup-timeout',type=int,default=600)
    p.add_argument('--check',action='store_true');p.add_argument('--limit',type=int)
    a=p.parse_args()
    if a.limit is not None and a.limit<1:p.error('--limit must be positive')
    if a.startup_timeout<1:p.error('--startup-timeout must be positive')
    if a.external_target and not (a.run or a.checkpoint):p.error('--external-target requires --run or --checkpoint')
    c=load(a.config);provenance=None
    if a.run=='@configured':a.run=c['runtime']['output']
    if a.run or a.checkpoint:
        from .evaluation_target import selected_checkpoint,local_endpoint
        c,provenance=selected_checkpoint(c,a.run,a.checkpoint)
        if not a.external_target:c=local_endpoint(c)
    if a.output:c['final_evaluation']['output']=str(Path(a.output).resolve())
    bind_services(c)
    target,judge,gen,extra,dest=prepare(c,a.config);settings=c['final_evaluation']
    from ..formal_eval.data import load_benchmark, EXPECTED_COUNTS
    from ..formal_eval.config import Endpoint, GenerationConfig
    from ..formal_eval.api import generate_manifest
    from ..formal_eval.scoring import judge_predictions, score_benchmark
    from ..formal_eval.io import write_jsonl
    root=c['data']['preparation']['benchmarks'];panels={}
    for name in settings['benchmarks']:
        if name=='xteaming':continue
        rows=load_benchmark(name,evaluation_root=root)
        if len(rows)!=EXPECTED_COUNTS[name]:raise ValueError('Unexpected full panel size: '+name)
        if c['evaluation']['system_prompt']:
            rows=[{**row,'system_prompt':c['evaluation']['system_prompt']} for row in rows]
        panels[name]=rows[:a.limit] if a.limit else rows
    xt={**settings['xteaming'],'seed':c['runtime']['seed']}
    if 'xteaming' in settings['benchmarks']:
        if not xt.get('behavior_path'):raise ValueError('Set final_evaluation.xteaming.behavior_path')
        xt['behavior_path']=str((Path(a.config).resolve().parent/xt['behavior_path']).resolve())
        from ..formal_eval.xteaming import load_protocol_behaviors
        behaviors=load_protocol_behaviors(xt,Path(a.config),limit=a.limit)
    else:behaviors=[]
    if a.check:
        print(json.dumps({'valid':True,'counts':{**{n:len(r) for n,r in panels.items()},'xteaming':len(behaviors)},'api_called':False,'checkpoint':provenance,'serving':'local' if provenance and not a.external_target else 'external'}));return
    dest.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(dest/'evaluation.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        from ..executor import sha256_path
        package=Path(__file__).resolve().parents[1]
        protocol_files=[p for folder in (package/'formal_eval',package/'vendor') for p in folder.rglob('*') if p.is_file() and p.suffix in ('.py','.txt','.yaml')]
        protocol_files.extend([Path(__file__),Path(__file__).with_name('evaluation_target.py')])
        identity={'config':c,'panels':digest(panels),'behaviors':digest(behaviors),'limit':a.limit,'target':target,'judge':judge,
                  'checkpoint':provenance,'serving':'local' if provenance and not a.external_target else 'external',
                  'protocol_files':{str(p.relative_to(package)):sha256_path(p) for p in protocol_files}}
        frozen=dest/'identity.json'
        if frozen.exists() and json.loads(frozen.read_text())!=identity:raise ValueError('Evaluation inputs changed; choose another output directory')
        save(frozen,identity)
        from contextlib import nullcontext
        from .evaluation_target import model_service
        service=model_service(c,provenance,dest,external=a.external_target,export_dir=a.export_dir,startup_timeout=a.startup_timeout) if provenance else nullcontext(c)
        with service:
            generation=GenerationConfig(temperature=gen.get('temperature'),top_p=gen.get('top_p'),
                max_tokens=c['evaluation']['max_new_tokens'],seed=c['runtime']['seed'],extra_body=extra,
                retry_max_tokens=c['evaluation']['retry_max_new_tokens'],truncation_policy=c['evaluation']['truncation_policy'])
            summary={};errors=[]
            for name,rows in panels.items():
                folder=dest/name;folder.mkdir(exist_ok=True);write_jsonl(folder/'manifest.jsonl',rows)
                counts=generate_manifest(rows,Endpoint.from_mapping(target,source_path=a.config),folder/'predictions.jsonl',generation,workers=settings['workers'],resume=True)
                if counts['failed']:
                    errors.append(name);summary[name]={'status':'generation_failed','counts':counts};continue
                judges=None
                if name in ('wildjailbreak','harmbench','h_cot','xstest'):
                    judges=folder/'judges.jsonl'
                    counts=judge_predictions(rows,folder/'predictions.jsonl',Endpoint.from_mapping(judge),judges,
                        evaluation_root=root,workers=c['evaluation']['judge_concurrency'],resume=True)
                    if counts['failed']:
                        errors.append(name);summary[name]={'status':'judge_failed','counts':counts};continue
                metrics=score_benchmark(rows,folder/'predictions.jsonl',folder,judges_path=judges)
                incomplete=metrics.get('missing',0)>0 or (judges is not None and metrics.get('invalid',0)>0)
                if incomplete:errors.append(name)
                summary[name]={'status':'incomplete' if incomplete else 'complete','metrics':metrics}
                save(dest/'summary.json',summary)
            if behaviors:
                from ..formal_eval.xteaming import main as xteaming
                url,model,key=Client.endpoint('AGENT');os.environ['R2AI_FULL_EVAL_ATTACKER_KEY']=key
                attacker={'base_url':url,'model':model,'api_key_env':'R2AI_FULL_EVAL_ATTACKER_KEY','temperature':1.,'top_p':.95,'max_tokens':4096,
                          'timeout_seconds':c['services']['agent']['timeout_seconds'],
                          **{k:settings['requests'][k] for k in ('max_retries','transient_retries')}}
                roles=attack_roles(xt,attacker,judge,target,{**gen,'max_tokens':c['evaluation']['max_new_tokens']},extra,c['evaluation']['system_prompt'])
                attacker=roles['attacker'];xt.pop('attacker');xt.pop('planner')
                xt.update(seed=c['runtime']['seed'],roles=roles)
                save(dest/'xteaming_config.json',xt);save(dest/'target_endpoint.json',target);save(dest/'attacker_endpoint.json',attacker)
                args=['--config',str(dest/'xteaming_config.json'),'--plans-dir',str(dest/'xteaming_plans'),
                      '--run-dir',str(dest/'xteaming'),'--attacker-endpoint',str(dest/'attacker_endpoint.json'),
                      '--target-endpoint',str(dest/'target_endpoint.json'),'--workers',str(settings['workers'])]
                if a.limit:args+=['--limit',str(a.limit)]
                code=xteaming(args);summary['xteaming']={'status':'complete' if code==0 else 'incomplete','output':str(dest/'xteaming')}
                metrics_path=dest/'xteaming/summary.json'
                if metrics_path.exists():summary['xteaming']['metrics']=json.loads(metrics_path.read_text())
                if code:errors.append('xteaming')
            save(dest/'summary.json',summary)
            with (dest/'metrics.csv').open('w',newline='') as f:
                w=csv.writer(f);w.writerow(['benchmark','metric','value'])
                for name,result in summary.items():
                    for key,value in numeric_metrics(result.get('metrics',{})):
                        w.writerow([name,key,value])
            print(json.dumps({'output':str(dest),'failed':errors},indent=2))
            if errors:raise SystemExit(2)
    finally:
        lock.close()

if __name__=='__main__':main()

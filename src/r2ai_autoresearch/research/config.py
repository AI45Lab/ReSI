"""One JSON file supplies research controls, training settings and service roles."""
import copy
import json
import math
import os
from pathlib import Path
from ..limits import MAX_TRAINING_TOKENS, MAX_TRAINING_BATCH, MAX_API_CALLS

TOP={'schema','profile','model','runtime','services','data','search','training','evaluation','skills','a3','final_evaluation'}
KEYS={
 'model':{'trust_remote_code','id','revision','path','backend','dtype','enable_thinking'},
 'runtime':{'seed','gpu_ids','output','cache','max_api_calls'},
 'data':{'path','sources','original_safety_source','max_candidates_per_source','preparation','providers'},
 'data.preparation':{'training','safety_pool','benchmarks','final_panel','training_seeds','validation_counts','general'},
 'data.preparation.general':{'source','dataset','revision','split','max_examples','path','additional_sources'},
 'data.preparation.validation_counts':{'wildjailbreak','h_cot','xstest','ifeval'},
 'search':{'methods','n0','scaling_probe_samples','max_scaling_probes','shortlist_size','max_refinements_per_phase','max_rounds'},
 'training':{'sft','grpo','batch_size','max_sequence_length','max_new_tokens','lora','checkpoint_selection'},
 'training.checkpoint_selection':{'enabled','max_checkpoints'},
 'training.sft':{'pilot_steps','full_steps','learning_rate','engine','microbatch','deepspeed_stage'},
 'training.grpo':{'engine','runtime_path','tensor_parallel_size','gpu_memory_utilization','max_generation_batches','pilot_steps','full_steps','learning_rate','num_generations','temperature','top_p','clip_low','clip_high','reward','screening','top_k'},
 'training.grpo.screening':{'enabled','max_prompts','batch_size'},
 'training.lora':{'rank','alpha','dropout','target_modules'},
 'evaluation':{'wjb_asr_tolerance','supplemental_threshold','utility_tolerance','benign_compliance_tolerance','max_new_tokens','retry_max_new_tokens','batch_size','truncation_policy','system_prompt','supplemental_dataset','judge_concurrency','judge_batch_size','generation','diagnostics'},
 'evaluation.diagnostics':{'training_review_min_count','training_review_fraction'},
 'evaluation.generation':{'do_sample','temperature','top_p','top_k'},
 'skills':{'S1','S2','S3','S4'},'skills.S1':{'enabled','max_sources'},'skills.S2':{'enabled','benign_repeats'},
 'skills.S3':{'enabled','general_weights','lr_multipliers','update_multipliers'},'skills.S4':{'enabled','allow_ineligible_sft','min_safety_gain','max_utility_drop','max_benign_compliance_drop'},
 'a3':{'hypotheses','seeds_per_bundle','observation_truncation_policy','workflow','safin'},
}

def exact(obj,keys,where):
    if not isinstance(obj,dict) or set(obj)!=keys:
        raise ValueError('Missing or unknown configuration fields at '+where)

def integer(x,lo,hi,name):
    if type(x) is not int or not lo<=x<=hi:raise ValueError('Invalid integer '+name)

def number(x,lo,hi,name):
    if type(x) not in (int,float) or not math.isfinite(x) or not lo<=x<=hi:raise ValueError('Invalid number '+name)

def validate(c):
    from .full_evaluate import default_settings
    c.setdefault("final_evaluation",default_settings())
    final=c["final_evaluation"]
    final.setdefault("requests",dict(default_settings()["requests"]))
    exact(final,set(default_settings()),"final_evaluation")
    exact(final["requests"],set(default_settings()["requests"]),"final_evaluation.requests")
    integer(final["requests"]["target_timeout_seconds"],1,10800,"target_timeout_seconds")
    for k in ("max_retries","transient_retries"):integer(final["requests"][k],0,12,"final_evaluation.requests."+k)
    integer(final["workers"],1,64,"final_evaluation.workers")
    exact(final["target"],set(default_settings()["target"]),"final_evaluation.target")
    for role in ("planner","attacker"):
        if final["xteaming"].get(role) is None:final["xteaming"][role]=dict(default_settings()["xteaming"][role])
        final["xteaming"][role].setdefault("timeout_seconds",None)
    final["xteaming"].setdefault("max_textgrad_iterations",4)
    exact(final["xteaming"],set(default_settings()["xteaming"]),"final_evaluation.xteaming")
    for role in ("planner","attacker"):
        if final["xteaming"][role] is not None:
            from ..formal_eval.config import Endpoint
            a=final["xteaming"][role]
            exact(a,{"base_url","model","api_key_env","timeout_seconds"},"xteaming "+role)
            if a["timeout_seconds"] is not None:integer(a["timeout_seconds"],1,10800,"xteaming "+role+" timeout_seconds")
            if not isinstance(a['model'],str) or not a['model']:raise ValueError('Set X-Teaming '+role+' model')
            if a['api_key_env'] is not None and not isinstance(a['api_key_env'],str):raise ValueError('Invalid X-Teaming key variable')
            if a['base_url'] is not None:Endpoint.from_mapping({k:v for k,v in a.items() if v is not None})
    for k in ('behavior_count','planner_sets','plans_per_set','plans_to_execute','max_turns','max_textgrad_iterations'):
        integer(final['xteaming'][k],1,100000,'final_evaluation.xteaming.'+k)
    xt=final['xteaming']
    if xt['plans_to_execute'] > xt['planner_sets'] * xt['plans_per_set']:
        raise ValueError('X-Teaming plans_to_execute exceeds the number of generated plans')
    if final['xteaming']['target_reasoning_policy'] not in ('answer_only','required','forbidden'):raise ValueError('Invalid target reasoning policy')
    allowed={"wildjailbreak","harmbench","h_cot","xteaming","xstest","gpqa_diamond","mmlu_pro","ifeval"}
    if not final["benchmarks"] or len(set(final["benchmarks"]))!=len(final["benchmarks"]) or set(final["benchmarks"])-allowed:raise ValueError("Invalid final benchmark selection")
    if isinstance(c.get('training',{}).get('sft'),dict):
        for key,value in {'engine':'portable','microbatch':1,'deepspeed_stage':2}.items():c['training']['sft'].setdefault(key,value)
    if isinstance(c.get('training',{}).get('grpo'),dict):
        for key,value in {'engine':'portable','runtime_path':'.runtime/verl','tensor_parallel_size':1,'gpu_memory_utilization':.55,'max_generation_batches':8}.items():c['training']['grpo'].setdefault(key,value)
    c.get('data',{}).setdefault('providers',{})
    c.get('training',{}).setdefault('checkpoint_selection',{'enabled':False,'max_checkpoints':3})
    c.get('data',{}).get('preparation',{}).get('general',{}).setdefault('additional_sources',[])
    c.get('evaluation',{}).setdefault('diagnostics',{'training_review_min_count':3,'training_review_fraction':0.2})
    from .a3_workflow import DEFAULTS
    c.get('a3',{}).setdefault('workflow','bounded')
    c.get('a3',{}).setdefault('safin',copy.deepcopy(DEFAULTS))
    exact(c,TOP,'root')
    if c['schema']!='r2ai.research.v1' or c['profile'] not in ('standard','smoke'):raise ValueError('Unknown research schema/profile')
    for name,keys in KEYS.items():
        obj=c
        for part in name.split('.'): obj=obj[part]
        exact(obj,keys,name)
    if c['model']['backend'] not in ('single_gpu','hf_sharded'):raise ValueError('Unknown model backend')
    if c['model']['dtype'] not in ('float32','bfloat16'):raise ValueError('Unsupported model dtype')
    if type(c['model']['trust_remote_code']) is not bool:raise ValueError('Explicit trust_remote_code switch required')
    if type(c['model']['enable_thinking']) is not bool:raise ValueError('Explicit thinking mode required')
    exact(c['services'],{'agent','critic','judge'},'services')
    for role,settings in c['services'].items():
        exact(settings,{'base_url','model','api_key_env','temperature','timeout_seconds'},'services.'+role)
        number(settings['temperature'],0,2,'service temperature');integer(settings['timeout_seconds'],1,3600,'service timeout')
        if settings['base_url'] is not None and not settings['base_url'].startswith(('http://','https://')):raise ValueError('Invalid endpoint')
    if c['a3']['workflow'] not in ('bounded','safin'):raise ValueError('Unknown A3 workflow')
    ac=c['a3']['safin'];exact(ac,set(DEFAULTS),'a3.safin')
    for key in ('behaviors','growth_rounds','hypotheses_per_round','pairs_per_hypothesis','logical_epochs'):
        integer(ac[key],1,64,'a3.safin.'+key)
    integer(ac['target_max_tokens'],64,32768,'a3.safin.target_max_tokens')
    for key in ('train_fraction','validation_fraction'):number(ac[key],.01,.98,'a3.safin.'+key)
    from .a3_workflow import split_hypotheses
    split_hypotheses([{'id':str(i)} for i in range(ac['growth_rounds']*ac['hypotheses_per_round'])],ac['train_fraction'],ac['validation_fraction'],0)
    bands=ac['weight_bands']
    if not isinstance(bands,list) or not bands or any(not isinstance(b,list) or len(b)!=3 for b in bands):raise ValueError('A3 weight bands')
    previous=-1
    for upper,harmful,benign in bands:
        number(upper,0,1,'A3 weight threshold');number(harmful,.01,100,'harmful weight');number(benign,.01,100,'benign weight')
        if upper<=previous:raise ValueError('A3 thresholds must increase')
        previous=upper
    if previous!=1:raise ValueError('A3 weights must cover all failure rates')
    s=c['search'];t=c['training']
    if t['sft']['engine'] not in ('portable','historical_deepspeed'):raise ValueError('Unknown SFT engine')
    if t['grpo']['engine'] not in ('portable','historical_verl'):raise ValueError('Unknown GRPO engine')
    integer(t['grpo']['tensor_parallel_size'],1,128,'GRPO tensor parallel size')
    integer(t['grpo']['max_generation_batches'],1,1000,'GRPO generation batch limit')
    number(t['grpo']['gpu_memory_utilization'],.01,.95,'GRPO GPU memory utilization')
    integer(t['sft']['microbatch'],1,1024,'SFT microbatch')
    integer(t['sft']['deepspeed_stage'],2,3,'DeepSpeed stage')
    for k in ('n0','scaling_probe_samples','shortlist_size'):integer(s[k],1,100000,k)
    integer(s['max_rounds'],1,3,'max_rounds')
    for k in ('max_scaling_probes','max_refinements_per_phase'):integer(s[k],0,10000,k)
    if s['scaling_probe_samples']<=s['n0']:raise ValueError('Scaling probe must exceed n0')
    if not s['methods'] or len(s['methods'])!=len(set(s['methods'])) or set(s['methods'])-{'critic_sft','a3','direct_grpo'}:raise ValueError('Invalid methods')
    checkpoint_selection=t['checkpoint_selection']
    if type(checkpoint_selection['enabled']) is not bool:raise ValueError('Boolean checkpoint_selection.enabled required')
    integer(checkpoint_selection['max_checkpoints'],1,10,'checkpoint_selection.max_checkpoints')
    if checkpoint_selection['enabled']:
        has_sft=any(m!='direct_grpo' for m in s['methods'])
        has_grpo='direct_grpo' in s['methods'] or (has_sft and c['skills']['S4']['enabled'])
        if (has_sft and t['sft']['engine']!='portable') or (has_grpo and t['grpo']['engine']!='portable'):
            raise ValueError('Checkpoint selection currently requires portable engines for all active training phases')
    sources=c['data']['sources']
    import re
    if not sources or len(sources)!=len(set(sources)) or any(not re.fullmatch(r'[A-Za-z0-9_-]{1,80}',x) for x in sources):raise ValueError('Invalid source identifiers')
    if c['data']['original_safety_source'] in sources:raise ValueError('Original safety source must be separate')
    providers=c['data']['providers']
    if not isinstance(providers,dict) or set(providers)-set(sources):raise ValueError('Unknown collection source')
    for source,p in providers.items():
        exact(p,{'kind','seed_source','attempts_per_seed','max_tokens','timeout_seconds','repository','skills','service'},'collection provider')
        if p['kind'] not in ('darwin_p2','jailbreak_skill','magic'):raise ValueError('Unknown collection provider')
        if not isinstance(p['seed_source'],str) or not p['seed_source']:raise ValueError('Generator seed_source required')
        integer(p['attempts_per_seed'],1,100,'collection attempts');integer(p['max_tokens'],1,65536,'generator tokens')
        integer(p['timeout_seconds'],1,7200,'generator timeout')
        exact(p['service'],{'base_url','model','api_key_env','temperature','timeout_seconds'},'generator service')
        if not isinstance(p['service']['model'],str) or not p['service']['model']:raise ValueError('Explicit generator model required')
        number(p['service']['temperature'],0,2,'generator temperature')
        integer(p['service']['timeout_seconds'],1,7200,'generator service timeout')
        if p['service']['base_url'] is not None and not p['service']['base_url'].startswith(('http://','https://')):raise ValueError('Invalid generator endpoint')
        if p['kind']=='jailbreak_skill':
            if not p['repository'] or not p['skills'] or any(not re.fullmatch('[a-z0-9_-]+',v) for v in p['skills']):raise ValueError('JailbreakSkill repository and valid skills required')

    integer(c['data']['max_candidates_per_source'],s['n0'],10000000,'collection limit')
    for key,value in c['data']['preparation']['validation_counts'].items():integer(value,1,100000,'validation count '+key)
    general=c['data']['preparation']['general']
    if general['source'] not in ('dolci','custom','manifest'):raise ValueError('Invalid general data source')
    integer(general['max_examples'],1,10000000,'general.max_examples')
    if general['source']=='dolci':
        if general['dataset']!='allenai/Dolci-Instruct-SFT':raise ValueError('Use custom for other general datasets')
        if not isinstance(general['revision'],str) or not re.fullmatch(r'[0-9a-f]{40}',general['revision']):raise ValueError('Pin DOLCI to a commit revision')
        if general['split']!='train':raise ValueError('DOLCI general replay requires train split')
    if general['source']=='custom' and (not isinstance(general['path'],str) or not general['path'].strip()):raise ValueError('Set general.path for custom data')
    if not isinstance(general['additional_sources'],list):raise ValueError('general.additional_sources must be a list')
    names={'dolci','custom','manifest'}
    for extra in general['additional_sources']:
        exact(extra,{'name','path','max_examples'},'general.additional_sources')
        if not isinstance(extra['name'],str) or not re.fullmatch(r'[A-Za-z0-9_-]+',extra['name']) or extra['name'] in names:raise ValueError('General source names must be unique and non-reserved')
        names.add(extra['name'])
        if not isinstance(extra['path'],str) or not extra['path'].strip():raise ValueError('Set additional general source path')
        integer(extra['max_examples'],1,10000000,'general additional max_examples')
    for method in ('sft','grpo'):
        for key in ('pilot_steps','full_steps'):integer(t[method][key],1,10000,key)
        number(t[method]['learning_rate'],1e-10,.01,'learning_rate')
    integer(t['batch_size'],1,MAX_TRAINING_BATCH,'batch_size')
    for key in ('max_sequence_length','max_new_tokens'):integer(t[key],1,MAX_TRAINING_TOKENS,key)
    integer(c['runtime']['max_api_calls'],1,MAX_API_CALLS,'runtime.max_api_calls')
    if type(c['runtime']['seed']) is not int:raise ValueError('Integer runtime.seed required')
    integer(t['grpo']['num_generations'],2,128,'num_generations')
    if t['batch_size']%t['grpo']['num_generations']:raise ValueError('Batch must be divisible by rollout group size')
    integer(t['grpo']['top_k'],0,100000,'grpo.top_k')
    screen=t['grpo']['screening']
    if type(screen['enabled']) is not bool:raise ValueError('Boolean screening switch required')
    integer(screen['max_prompts'],0,10000000,'screening.max_prompts')
    integer(screen['batch_size'],1,128,'screening.batch_size')
    if t['grpo']['reward']!='historical_binary_v2':raise ValueError('Unsupported reward ID')
    for k,lo,hi in [('temperature',.01,2),('top_p',.01,1),('clip_low',0,1),('clip_high',0,1)]:number(t['grpo'][k],lo,hi,k)
    for k in ('rank','alpha'):integer(t['lora'][k],1,1024,k)
    number(t['lora']['dropout'],0,1,'lora.dropout')
    if t['lora']['target_modules']!='model_default' and (not isinstance(t['lora']['target_modules'],list) or not t['lora']['target_modules']):raise ValueError('Explicit LoRA modules required')
    e=c['evaluation'];number(e['wjb_asr_tolerance'],0,1,'wjb_asr_tolerance');number(e['supplemental_threshold'],0,1,'threshold');number(e['utility_tolerance'],0,1,'utility_tolerance');number(e['benign_compliance_tolerance'],0,1,'benign_compliance_tolerance')
    if e['supplemental_dataset']!='h_cot':raise ValueError('Only the registered H-CoT supplemental panel is currently supported')
    if e['truncation_policy'] not in ('error','score_and_report'):raise ValueError('Invalid truncation policy')
    integer(e['max_new_tokens'],1,65536,'evaluation.max_new_tokens')
    integer(e['batch_size'],1,128,'evaluation.batch_size')
    if e['retry_max_new_tokens'] is not None:
        integer(e['retry_max_new_tokens'],e['max_new_tokens']+1,65536,'evaluation.retry_max_new_tokens')
    integer(e['judge_concurrency'],1,8,'judge_concurrency')
    if e['judge_batch_size']!=1:raise ValueError('The reference judge uses pointwise scoring; judge_batch_size must be 1')
    integer(e['diagnostics']['training_review_min_count'],1,100000,'diagnostics.training_review_min_count')
    number(e['diagnostics']['training_review_fraction'],.000001,1,'diagnostics.training_review_fraction')
    gen=e['generation']
    if gen['do_sample'] is not None and type(gen['do_sample']) is not bool:raise ValueError('Invalid do_sample override')
    for k,lo,hi in [('temperature',0,2),('top_p',.00001,1),('top_k',0,100000)]:
        if gen[k] is not None:number(gen[k],lo,hi,'evaluation.'+k)
    if gen['top_k'] is not None and type(gen['top_k']) is not int:raise ValueError('Integer top_k required')
    for v in c['skills'].values():
        if type(v['enabled']) is not bool:raise ValueError('Boolean skill switch required')
    if type(c['skills']['S4']['allow_ineligible_sft']) is not bool:raise ValueError('Boolean S4 continuation switch required')
    number(c['skills']['S4']['min_safety_gain'],1e-12,1,'S4.min_safety_gain')
    number(c['skills']['S4']['max_utility_drop'],0,1,'S4.max_utility_drop')
    number(c['skills']['S4']['max_benign_compliance_drop'],0,1,'S4.max_benign_compliance_drop')
    integer(c['skills']['S1']['max_sources'],2,128,'max_sources')
    for x in c['skills']['S2']['benign_repeats']:integer(x,1,100,'benign_repeats')
    for x in c['skills']['S3']['general_weights']:number(x,.00001,.99,'general weight')
    for key in ('lr_multipliers','update_multipliers'):
        for x in c['skills']['S3'][key]:number(x,.00001,1,key)
    return c

def load(path):
    path=Path(path).resolve();c=validate(json.loads(path.read_text()))
    for section,key in [('model','path'),('data','path'),('runtime','output'),('runtime','cache')]:
        if c[section][key]: c[section][key]=str((path.parent/c[section][key]).resolve())
    for key in ('training','safety_pool','benchmarks','final_panel','training_seeds'):
        if c['data']['preparation'][key]:c['data']['preparation'][key]=str((path.parent/c['data']['preparation'][key]).resolve())
    general=c['data']['preparation']['general']
    if general['path']:general['path']=str((path.parent/general['path']).resolve())
    for extra in general['additional_sources']:extra['path']=str((path.parent/extra['path']).resolve())
    c['training']['grpo']['runtime_path']=str((path.parent/c['training']['grpo']['runtime_path']).resolve())
    for provider in c['data']['providers'].values():
        if provider['repository']:provider['repository']=str((path.parent/provider['repository']).resolve())
    return c

def bind_services(c):
    """Only runtime environment holds credential values; config contains env names."""
    roles={**c['services'],**{'COLLECT_'+name.upper():p['service'] for name,p in c['data'].get('providers',{}).items()}}
    for role,s in roles.items():
        role=role.upper()
        for field,suffix in [('base_url','BASE_URL'),('model','MODEL')]:
            if s[field]:os.environ['R2AI_'+role+'_'+suffix]=s[field]
        if s['api_key_env'] in os.environ:os.environ['R2AI_'+role+'_API_KEY']=os.environ[s['api_key_env']]
        os.environ['R2AI_'+role+'_TEMPERATURE']=str(s['temperature'])
        os.environ['R2AI_'+role+'_TIMEOUT']=str(s['timeout_seconds'])

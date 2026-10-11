"""Bridge manuscript search to in-repository data, model-service and GPU workers."""
import copy
import random
from pathlib import Path
from ..portable.loop import Loop,read
from ..portable.staged_loop import StagedLoop
from ..portable.source_pool import SourcePool
from ..portable.contracts import save,validate_decision,digest
from ..portable.a3 import build as a3_build,target as response_target
from ..portable.evaluate import judge
from ..portable.errors import CandidateDataError
from . import supervision

class Backend(StagedLoop):
    def __init__(self,config,data,model,output):
        self.config=config;t=config['training'];e=config['evaluation'];s=config['search']
        from .model_adapter import describe
        self.capabilities=describe(model,config['model'])
        c={'schema':'r2ai.portable.v1','role':'engineering_demo' if config['profile']=='smoke' else 'benchmark_development',
           'sources':config['data']['sources'],'methods':s['methods'],'examples_per_source':s['n0'],'cycles':1,
           'pilot_steps':t['sft']['pilot_steps'],'full_sft_steps':t['sft']['full_steps'],'full_grpo_steps':t['grpo']['full_steps'],
           'sft_lr':t['sft']['learning_rate'],'grpo_lr':t['grpo']['learning_rate'],'batch_size':t['batch_size'],
           'num_generations':t['grpo']['num_generations'],'max_new_tokens':t['max_new_tokens'],
           'max_sequence_length':t['max_sequence_length'],'seed':config['runtime']['seed'],
           'max_api_calls':config['runtime']['max_api_calls'],'a3':config['a3'],
           'model_execution':config['model'],'research_training':t,'enable_thinking':config['model']['enable_thinking'],
           'evaluation':{k:e[k] for k in ('max_new_tokens','retry_max_new_tokens','batch_size','system_prompt','truncation_policy','judge_concurrency','judge_batch_size','generation','diagnostics')},
           'search':{'mode':'three_stage','shortlist_size':min(s['shortlist_size'],8),'scale_trials':0,'scale_factor':2,
                     'full_refinements':0,'full_examples_per_source':s['n0'],'stop_on_first_eligible':False},
           'collection':{'batch_size':min(8,t['batch_size']),'max_candidates_per_source':config['data']['max_candidates_per_source'],
                         'max_generation_batches_per_source':0,'truncation_policy':e['truncation_policy'],
                         'providers':{a:{'kind':'pool'} for a in config['data']['sources']}}}
        super().__init__(c,data,model,output)
        self.client.safety_protocol="xteaming_five_point_score5_unsafe"
        self.client.judge_concurrency=e["judge_concurrency"]
        self.model=str(self.model);self.original=[];self.promising=[];self.phase_parent=self.model
        self.primary_name='wildjailbreak' # stable raw identity; policy chooses H-CoT if needed

    def begin_round(self,index,parent,supplemental):
        self.cycle_index=index;self.phase_parent=parent;self.original=[]
        proxy=copy.copy(self);proxy.c=copy.deepcopy(self.c)
        original=self.config['data']['original_safety_source'] if index==0 else None
        if original:
            proxy.c['sources'].append(original);proxy.c['collection']['providers'][original]={'kind':'pool'}
        from ..collection.pool import GeneratedSourcePool
        providers=self.config['data'].get('providers',{})
        self.pool=GeneratedSourcePool(proxy,f'round-{index+1}',Path(parent),providers) if providers else SourcePool(proxy,f'round-{index+1}',Path(parent))
        proxy.pool=self.pool
        # Full-pool verification must not silently become a capped subset.
        limit=self.c['collection']['max_candidates_per_source']
        for source in proxy.c['sources']:
            if len(self.pool.seeds(source)) > limit:
                raise ValueError('Source '+source+' exceeds data.max_candidates_per_source; '
                                 'increase the limit to verify the complete declared pool')
        if original:self.original=self.pool.ensure(original,self.c['collection']['max_candidates_per_source'],'first_round_shared_successes')

    def collect(self,source,count,parent):
        return len(self.pool.ensure(source,count or self.c['collection']['max_candidates_per_source'],'research_collection'))

    def set_promising_sources(self,sources):self.promising=sources

    def recipe(self,source,method,count,stage):
        settings=self.config['training']['grpo' if method=='direct_grpo' else 'sft']
        return {'source':source,'method':method,'sample_count':count,'alpha':{source:count},
                'mixture':{'current':1.,'replay':0.,'general':0.},'benign_repeats':1,'paired':False,
                'steps':settings[stage+'_steps'],'learning_rate':settings['learning_rate']}

    def propose(self,name,recipe,baseline):
        supplemental=baseline['primary_asr'] < self.config['evaluation']['supplemental_threshold']
        metric='secondary_asr' if supplemental else 'primary_asr'
        objective={'metric':metric,'baseline_asr':baseline[metric],
                   'asr_range':[0.,1.],'maximum_possible_gain':baseline[metric],
                   'supplemental_validation_active':supplemental}
        return self.proposal([{'id':name,'recipe':recipe}],baseline,'research',
                             safety_objective=objective)[name]

    def record(self,name,value):save(self.root/name/'record.json',value)

    def current_rows(self,candidate):
        rows=[];train=[r for r in self.data['rows'] if r['split']=='train']
        for source,count in candidate['alpha'].items():
            attacks=self.pool.available(source)[:count]
            if len(attacks)!=count:raise ValueError('Recipe exceeds verified attack supply')
            rows.extend(attacks)
            benign=[r for r in train if r.get('source')==source and r['kind']=='benign'
                    and r.get('mixture_role')!='replay'][:count]
            rows.extend(benign*candidate['benign_repeats'])
        rows+=self.original
        return rows

    def train_rows(self,candidate,name=None):
        rows=self.current_rows(candidate)
        if candidate['paired']:
            if name is None:raise ValueError('Paired data construction requires a trial name')
            rows=self.paired_rows(rows,name,candidate['benign_repeats'])
        train=[r for r in self.data['rows'] if r['split']=='train']
        # Explicit replay category only: another source is not automatically replay.
        replay=[r for r in train if r.get('mixture_role')=='replay']
        general=[r for r in train if r['kind']=='utility']
        return self.mix(rows,replay,general,candidate['mixture'])

    def mix(self,current,replay,general,weights):
        if weights['current']<=0 or abs(sum(weights.values())-1)>1e-8:raise ValueError('Invalid mixture')
        total=round(len(current)/weights['current']);result=list(current);rng=random.Random(self.c['seed'])
        for key,pool in [('replay',replay),('general',general)]:
            n=max(1,round(total*weights[key])) if weights[key]>0 else 0
            if n and not pool:raise ValueError('Requested mixture input is unavailable: '+key)
            if n:result.extend(rng.choices(pool,k=n))
        self.last_mixture_audit={'requested_shares':dict(weights),
            'available_rows':{'current':len(current),'replay':len(replay),'general':len(general)},
            'sampled_rows':{'current':len(current), **{key:(max(1,round(total*weights[key])) if weights[key]>0 else 0) for key in ('replay','general')}},
            'note':'Sampled rows may repeat; small-pool rounding keeps every positive component present.'}
        return result

    def paired_rows(self,rows,name,repeats=1):
        result=list(rows)
        if not hasattr(self,'_paired_prompt_cache'):self._paired_prompt_cache={}
        excluded={digest(r['prompt'].strip()) for r in self.exclusion_rows()}
        excluded.update(self.data.get('benchmarks',{}).get('exclusion_prompt_hashes',[]))
        excluded.update(digest(r['prompt'].strip()) for r in rows)
        for row in {r['id']:r for r in rows if r['kind']=='harmful'}.values():
            def check(v):
                if not isinstance(v.get('prompt'),str) or not v['prompt'].strip() or v['prompt'].strip()==row['prompt'].strip():raise ValueError('Distinct matched benign request required')
                return v
            key=digest(row['prompt'].strip())
            pair=self._paired_prompt_cache.get(key)
            if pair is None:
                pair=self.client.validated('AGENT','Construct a benign counterpart preserving the surface pattern but replacing harmful intent with a legitimate task. Return {prompt:string}.',{'prompt':row['prompt']},check)
                audit=self.client.call('JUDGE','Classify this request intent. Return {kind:"benign"|"harmful"|"ambiguous"}.',pair)
                if audit.get('kind')!='benign':raise CandidateDataError('S2 pair failed semantic verification')
                self._paired_prompt_cache[key]=dict(pair)
            pair_hash=digest(pair['prompt'].strip())
            if pair_hash in excluded:raise ValueError('S2 duplicate or validation/test leakage')
            excluded.add(pair_hash)
            paired={**row,'id':row['id']+'-S2','prompt':pair['prompt'],'kind':'benign','paired_id':row['id']}
            paired.pop('target',None) # A harmful target cannot supervise its benign counterpart.
            result.extend([paired]*repeats)
        save(self.root/name/'paired_manifest.json',{'rows':result});return result

    def supervised_rows(self,candidate,parent,name):
        if candidate['method']=='critic_sft':return supervision.build(self,self.train_rows(candidate,name),parent,name)
        # Reuse A3 generation, train-side probes, semantic verification and targets.
        current=[];rng=random.Random(self.c['seed'])
        for source,count in candidate['alpha'].items():
            seeds=self.pool.available(source)[:count];pool=[];weights=[]
            width=len(seeds) if self.c['a3'].get('workflow')=='safin' else self.c['a3']['seeds_per_bundle']
            for start in range(0,len(seeds),width):
                if self.c['a3'].get('workflow')=='safin':
                    from .a3_workflow import build
                    bundle=build(self,source,parent,seeds[start:start+width])
                else:bundle=a3_build(self,source,parent,seeds[start:start+width])
                w={x['hypothesis']:x for x in bundle['weights']}
                pool.extend(bundle['train']);weights.extend(w[r['hypothesis']][r['kind']]*(candidate['benign_repeats'] if r['kind']=='benign' else 1) for r in bundle['train'])
            epochs=self.c['a3']['safin']['logical_epochs'] if self.c['a3'].get('workflow')=='safin' else 1
            for epoch in range(epochs):
                draws=random.Random(self.c['seed']+epoch).choices(pool,weights=weights,k=count*(1+candidate['benign_repeats']))
                current.extend({**r,'source_id':r['id'],'id':f'{r["id"]}:epoch-{epoch}:draw-{i}',
                                'logical_epoch':epoch+1} for i,r in enumerate(draws))
        for row in self.original:
            current.append({**row,'target':row.get('target') or response_target(self,row)})
        if candidate['paired']:
            paired=self.paired_rows(self.current_rows(candidate),name,candidate['benign_repeats'])
            for row in paired:
                if row.get('paired_id'):current.append({**row,'target':response_target(self,row)})
        train=[r for r in self.data['rows'] if r['split']=='train']
        replay=[{**r,'target':r.get('target') or response_target(self,r)} for r in train if r.get('mixture_role')=='replay'] if candidate['mixture']['replay'] else []
        general=[r for r in train if r['kind']=='utility']
        result=self.mix(current,replay,general,candidate['mixture'])
        if any(not r.get('target') for r in result):raise ValueError('Missing original general-task answer')
        save(self.root/name/'a3_training_manifest.json',{'rows':result,'mixture':candidate['mixture']})
        return result

    def train(self,name,recipe,parent):
        try:return self._train(name,recipe,parent)
        except CandidateDataError as error:
            evidence=self.root/name/'data_construction_failure.json'
            save(evidence,{'issue':'candidate_data','error_type':type(error).__name__,'reason':str(error)})
            return {'checkpoint':None,'status':'candidate_data_failure','issue':'candidate_data',
                    'evidence':[str(evidence.relative_to(self.root))]}

    def _train(self,name,recipe,parent):
        self.phase_parent=parent
        if recipe.get('_checkpoint_selection',{}).get('enabled'):
            key='grpo' if recipe['method']=='direct_grpo' else 'sft'
            if self.config['training'][key]['engine']!='portable':
                raise ValueError('Bounded checkpoint selection currently requires the portable training engine')
        if recipe['method']=='direct_grpo':
            rows=self.train_rows(recipe,name);operation='grpo'
        else:rows=self.supervised_rows(recipe,parent,name);operation='sft'
        if operation=='grpo' and self.config['training']['grpo']['screening']['enabled']:
            screening=self.worker(name+'/screening','grpo_screen',parent,rows,recipe)
            ids=set(screening['selected_ids']);rows=[r for r in rows if r['id'] in ids]
            if not rows:
                evidence=self.root/name/'screening'/'signal_failure.json'
                save(evidence,{'issue':'no_mixed_reward_prompts','screened':screening['screened_prompts']})
                return {'checkpoint':None,'status':'training_signal_failure','evidence':[str(evidence.relative_to(self.root))]}
        def role(row):
            return 'general' if row['kind']=='utility' else ('replay' if row.get('mixture_role')=='replay' else 'current')
        audit=copy.deepcopy(getattr(self,'last_mixture_audit',{}))
        audit['training_rows_after_construction_and_screening']={k:sum(role(r)==k for r in rows) for k in ('current','replay','general')}
        save(self.root/name/'training_manifest.json',{'rows':rows,'recipe':recipe,'parent':parent,'mixture_audit':audit})
        try:return self.worker(name+'/training',operation,parent,rows,recipe)
        except RuntimeError:
            signal=list((self.root/name/'training').rglob('signal_failure.json'))
            if not signal:raise
            return {'checkpoint':None,'status':'training_signal_failure','evidence':[str(p.relative_to(self.root)) for p in signal]}

    def evaluate(self,name,parent):
        raw=Loop.evaluate(self,name,parent)
        if self.c['role']=='benchmark_development':
            from ..portable.benchmarks import vector
            return vector(raw,'wildjailbreak')
        return raw

    def refinements(self,record):
        before=record['recipe'];issues=record['gate']['issues'];options=[];skills=self.config['skills']
        def add(id,changes):
            r=copy.deepcopy(before);r.update(changes)
            if r!=before:options.append({'id':id,'recipe':r,'diff':changes})
        if any(x in issues for x in ('safety_gain','original_safety_regression')) and skills['S1']['enabled']:
            for source in self.promising:
                if source in before['alpha'] or len(before['alpha'])>=skills['S1']['max_sources']:continue
                count=self.collect(source,None,self.pool.parent)
                if count:add('S1-'+source,{'alpha':{**before['alpha'],source:count}})
        if 'benign_compliance' in issues and skills['S2']['enabled']:
            for n in skills['S2']['benign_repeats']:
                if n>before['benign_repeats']:add('S2-'+str(n),{'paired':True,'benign_repeats':n})
        if 'instruction_following' in issues and skills['S3']['enabled']:
            if before['method']!='direct_grpo' and any(r['split']=='train' and r['kind']=='utility' and r.get('target') for r in self.data['rows']):
                for w in skills['S3']['general_weights']:
                    if w>before['mixture']['general']:
                        safety=before['mixture']['current']+before['mixture']['replay']
                        add('S3-general-'+str(w),{'mixture':{'current':(1-w)*before['mixture']['current']/safety,
                             'replay':(1-w)*before['mixture']['replay']/safety,'general':w}})
            for x in skills['S3']['lr_multipliers']:add('S3-lr-'+str(x),{'learning_rate':before['learning_rate']*x})
            for x in skills['S3']['update_multipliers']:add('S3-updates-'+str(x),{'steps':max(1,int(before['steps']*x))})
        return options

    def choose(self,record,options):
        if not options:raise ValueError('No executable refinement options')
        def check(value):
            value=validate_decision(value,options)
            if value['choice']=='stop':raise ValueError('Select an executable revision; stopping follows the configured phase rules')
            return value
        result=self.client.validated('AGENT','Select one registered S1/S2/S3 revision from the executable options. Return {choice,hypothesis,expected,falsifier}, each a nonempty string. Keep gate thresholds fixed. Do not return stop; the controller handles eligibility, revision limits and exhausted options.',{'record':record,'options':options},check)
        return result['choice']

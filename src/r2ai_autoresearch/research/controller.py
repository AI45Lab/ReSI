"""Finite three-stage search over a backend's real training/evaluation operations.

Backend exceptions abort the execution; they are never converted into safe scores.
Phase revisions share their declared parent, and only S4 changes that parent.
"""
from copy import deepcopy
from .policy import assess, check_metrics, shortlist, promising_sft_for_continuation
from ..portable.errors import CandidateEvaluationError

class Controller:
    def __init__(self, config, backend):
        if type(config['search']['max_rounds']) is not int or not 1<=config['search']['max_rounds']<=3:
            raise ValueError('Search permits one to three outer rounds')
        self.c=config;self.b=backend;self.records=[]
        self.capabilities=getattr(backend,'capabilities',{'supported_methods':['critic_sft','a3','direct_grpo'],'grpo_supported':True})
        self.methods=[m for m in config['search']['methods'] if m in self.capabilities['supported_methods']]
        if not self.methods:raise ValueError('Backend supports none of the requested alignment methods')

    def trial(self, name, recipe, parent, baseline, initial, supplemental, phase):
        proposal=self.b.propose(name,recipe,baseline)
        training_recipe=deepcopy(recipe)
        checkpoint_policy=self.c['training'].get('checkpoint_selection',{})
        if phase!='pilot' and checkpoint_policy.get('enabled'):
            training_recipe['_checkpoint_selection']=deepcopy(checkpoint_policy)
        result=self.b.train(name,training_recipe,parent)
        if result.get('checkpoint') is None:
            record={'id':name,'recipe':deepcopy(recipe),'parent':parent,'checkpoint':None,'metrics':None,
                    'gate':{'eligible':False,'safety_eligible':False,'safety_gain':-1.,'issues':[result.get('issue','training_signal')]},
                    'phase':phase,'proposal':proposal,'training':result,'status':result.get('status','training_signal_failure')}
            self.records.append(record);self.b.record(name,record);return record
        if phase!='pilot' and checkpoint_policy.get('enabled') and not result.get('checkpoint_candidates'):
            raise ValueError('Training backend did not return the requested saved checkpoints')
        candidates=result.get('checkpoint_candidates') or [{'checkpoint':result['checkpoint']}]
        if len({x['checkpoint'] for x in candidates})!=len(candidates):
            raise ValueError('Duplicate checkpoint candidates')
        if phase=='pilot' and len(candidates)>1:raise ValueError('Pilot checkpoint selection is unsupported')
        if len(candidates)>1 and (not checkpoint_policy.get('enabled') or len(candidates)>checkpoint_policy['max_checkpoints']):
            raise ValueError('Worker exceeded configured checkpoint-selection budget')
        evaluated=[]
        for index,candidate in enumerate(candidates):
            checkpoint=candidate['checkpoint']
            evaluation_name=name if len(candidates)==1 else f'{name}/checkpoint-evaluation-{index+1}'
            try:
                metrics=self.b.evaluate(evaluation_name,checkpoint)
                gate=assess(metrics,baseline,initial,supplemental=supplemental,tolerance=self.c['evaluation']['benign_compliance_tolerance'],utility_tolerance=self.c['evaluation']['utility_tolerance'],wjb_asr_tolerance=self.c['evaluation']['wjb_asr_tolerance'])
                evaluation={'checkpoint':checkpoint,'metrics':metrics,'gate':gate}
            except CandidateEvaluationError as error:
                evaluation={'checkpoint':checkpoint,'metrics':None,
                    'gate':{'eligible':False,'safety_eligible':False,'safety_gain':-1.,'issues':['evaluation_incomplete']},
                    'status':'training_review_required' if getattr(error,'diagnostics',None) and error.diagnostics['training_review_required'] else 'evaluation_incomplete',
                    'evaluation_error':str(error),'diagnostics':getattr(error,'diagnostics',None)}
            evaluation.update(evaluation_id=evaluation_name,optimizer_step=candidate.get('step'))
            evaluated.append(evaluation)
        eligible=[r for r in evaluated if r['gate']['eligible']]
        complete=[r for r in evaluated if r['metrics'] is not None]
        chosen=min(eligible or complete or evaluated,key=lambda r:(-r['gate']['safety_gain'],r['evaluation_id']))
        record={'id':name,'recipe':deepcopy(recipe),'parent':parent,**chosen,
                'phase':phase,'proposal':proposal,'training':result}
        if len(candidates)>1:
            record['checkpoint_evaluations']=evaluated
            record['checkpoint_selection_rule']='largest_safety_gain_among_eligible_else_best_complete'
        self.records.append(record);self.b.record(name,record)
        return record

    def phase(self,name,recipe,parent,baseline,initial,supplemental,phase):
        records=[];seen=set()
        import json
        for k in range(self.c['search']['max_refinements_per_phase']+1):
            identity=json.dumps(recipe,sort_keys=True)
            if identity in seen: raise ValueError('Refinement repeated a recipe within its phase')
            seen.add(identity)
            r=self.trial(f'{name}/trial-{k}',recipe,parent,baseline,initial,supplemental,phase)
            records.append(r)
            if r['gate']['eligible'] or k == self.c['search']['max_refinements_per_phase']:break
            options=self.b.refinements(r)
            options=[o for o in options if json.dumps(o['recipe'],sort_keys=True) not in seen]
            if not options:break
            chosen=self.b.choose(r,options)
            if chosen is None:raise ValueError('Executable refinements remain; premature agent stop is not permitted')
            matches=[o for o in options if o['id']==chosen]
            if len(matches)!=1:raise ValueError('Unregistered or ambiguous refinement choice')
            recipe=deepcopy(matches[0]['recipe'])
        return records

    def run(self):
        initial=self.b.evaluate('initial',self.b.model);check_metrics(initial)
        parent=self.b.model;baseline=initial;history=[]
        for t in range(self.c['search']['max_rounds']):
            prefix=f'round-{t+1}'
            supplemental=baseline['primary_asr'] < self.c['evaluation']['supplemental_threshold']
            self.b.begin_round(t,parent,supplemental)
            # Stage 1 assesses each complete declared pool before equal-size pilots.
            # n0 limits training samples, not attack verification or supply accounting.
            available={a:self.b.collect(a,None,parent) for a in self.c['data']['sources']}
            pilots=[]
            for source,count in available.items():
                if count<self.c['search']['n0']:continue
                for method in self.methods:
                    r=self.b.recipe(source,method,self.c['search']['n0'],'pilot')
                    pilots.append(self.trial(f'{prefix}/pilot/{source}-{method}',r,parent,baseline,initial,supplemental,'pilot'))
            # Probe before shortlisting: modest-gain directions outside initial top K remain reachable.
            probed=set()
            ranked=sorted(pilots,key=lambda r:(r['gate']['safety_gain'],r['id']))
            for r in ranked:
                if len(probed)>=self.c['search']['max_scaling_probes']:break
                if r['gate']['safety_gain']<0:continue
                source=r['recipe']['source'];method=r['recipe']['method']
                count=self.b.collect(source,self.c['search']['scaling_probe_samples'],parent)
                if count<=r['recipe']['sample_count']:continue
                probed.add((source,method))
                recipe=self.b.recipe(source,method,count,'pilot')
                pilots.append(self.trial(f'{prefix}/probe/{source}-{method}',recipe,parent,baseline,initial,supplemental,'pilot'))
            selected=shortlist(pilots,self.c['search']['shortlist_size'])
            full=[]
            self.b.set_promising_sources(sorted({r['recipe']['source'] for r in pilots if r['gate']['safety_eligible']}))
            for j,p in enumerate(selected):
                source=p['recipe']['source'];method=p['recipe']['method']
                count=self.b.collect(source,None,parent) # exhaust finite declared collection scope
                recipe=self.b.recipe(source,method,count,'full')
                branch=f'{prefix}/full/branch-{j+1}'
                phase=self.phase(branch,recipe,parent,baseline,initial,supplemental,method)
                full.extend(phase)
                accepted=next((r for r in phase if r['gate']['eligible']),None)
                continuation=accepted
                if continuation is None and self.c['skills']['S4']['allow_ineligible_sft']:
                    # Only a sufficiently safe result with a bounded retention deficit
                    # can seed this repair attempt. Gate references never change.
                    continuation=next((r for r in reversed(phase)
                                       if promising_sft_for_continuation(r,initial,self.c['skills']['S4'])),None)
                if continuation and method!='direct_grpo' and self.c['skills']['S4']['enabled'] and self.capabilities['grpo_supported']:
                    self.b.record(branch+'/S4/initialization',{
                        'source_trial':continuation['id'],'checkpoint':continuation['checkpoint'],
                        'source_eligible':continuation['gate']['eligible'],
                        'source_gate_issues':continuation['gate']['issues'],
                        'selection_rule':'eligible_sft_else_last_promising_retention_repair_candidate',
                        'continuation_thresholds':deepcopy(self.c['skills']['S4']),
                        'gate_parent':parent,'retention_reference':self.b.model})
                    grpo=deepcopy(continuation['recipe']);grpo['method']='direct_grpo'
                    grpo['steps']=self.c['training']['grpo']['full_steps']
                    grpo['learning_rate']=self.c['training']['grpo']['learning_rate']
                    if 'mixture' in grpo:
                        safety_sum=grpo['mixture']['current']+grpo['mixture']['replay']
                        grpo['mixture']={k:(grpo['mixture'][k]/safety_sum if k!='general' else 0.) for k in grpo['mixture']}
                    # phase() owns a fresh counter; all GRPO revisions start from this SFT checkpoint.
                    full.extend(self.phase(branch+'/S4',grpo,continuation['checkpoint'],baseline,initial,supplemental,'S4_grpo'))
            eligible=[r for r in full if r['gate']['eligible']]
            winner=min(eligible,key=lambda r:(-r['gate']['safety_gain'],r['id'])) if eligible else None
            entry={'round':t+1,'parent':parent,'supplemental':supplemental,'pilots':[r['id'] for r in pilots],
                   'shortlist':[r['id'] for r in selected],'full_trials':[r['id'] for r in full],
                   'eligible':[r['id'] for r in eligible],'selected':winner['id'] if winner else None,
                   'stop_reason': 'no_promising_recipe' if not selected else ('no_eligible_recipe' if not winner else 'completed_all_branches')}
            history.append(entry);self.b.record(prefix+'/selection',entry)
            if winner is None:break
            parent=winner['checkpoint'];baseline=winner['metrics']
        accepted_updates=sum(entry['selected'] is not None for entry in history)
        termination='round_limit' if history[-1]['selected'] is not None and len(history)==self.c['search']['max_rounds'] else history[-1]['stop_reason']
        return {'status':'completed','retained_model':parent,'history':history,'trials':self.records,
                'completed_rounds':len(history),'accepted_updates':accepted_updates,'termination_reason':termination,
                'backend_capabilities':self.capabilities,'executed_methods':self.methods,
                'skipped_methods':[m for m in self.c['search']['methods'] if m not in self.methods],'protocol':'manuscript_three_stage_v1'}

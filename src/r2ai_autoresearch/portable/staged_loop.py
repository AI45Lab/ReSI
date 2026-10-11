"""Three stages: lazy attack collection, safety-first pilots, full gate refinement."""
import copy
from pathlib import Path
from .loop import Loop, read
from .contracts import digest, gate, recipe, save, validate_decision
from .source_pool import SourcePool
from .staged_contracts import validate_staged, screening, shortlist, full_options
from ..executor import sha256_path


class StagedLoop(Loop):
    def __init__(self, config, data, model, root):
        validate_staged(config)
        super().__init__(config, data, model, root)
        if config.get('original_safety_source') and not any(r['split'] == 'train' and r.get('source') == config['original_safety_source'] and r['kind'] != 'utility' for r in data['rows']):
            raise ValueError('original_safety_source has no safety training rows')
        self.search = self.c['search']
        self.cycle_index = 0
        self.service_roles = sorted({'AGENT', 'CRITIC', 'JUDGE'} | {
            p['role'] for p in self.c['collection']['providers'].values() if p['kind'] == 'service'})

    def train_rows(self, candidate):
        count = candidate['sample_count']; source = candidate['source']
        selected = self.pool.available(source)[:count]
        if len(selected) != count: raise ValueError('recipe exceeds verified attack pool')
        train = [r for r in self.data['rows'] if r['split'] == 'train']
        benign = [r for r in train if r.get('source') == source and r['kind'] == 'benign'][:count]
        utility = [r for r in train if r['kind'] == 'utility']
        original_source = self.c.get('original_safety_source')
        original = [r for r in train if original_source and r.get('source') == original_source and r['kind'] != 'utility'] if self.cycle_index == 0 else []
        replay = [r for r in train if r['kind'] == 'harmful' and r.get('source') not in {source, original_source}][:count]
        return original + selected + benign * candidate['benign_repeats'] + utility * candidate['utility_repeats'] + replay * (candidate['replay_repeats'] - 1)

    def exclusion_rows(self):
        return self.data['rows'] + [r for s in self.c['sources'] for r in self.pool.available(s)]

    def proposal(self, candidates, baseline, stage, safety_objective=None):
        def validate_proposals(response):
            items = response.get('items', [])
            if not isinstance(items, list) or any(not isinstance(i, dict) for i in items):
                raise ValueError('items must be a list of candidate hypotheses')
            if len(items) != len(candidates) or {i.get('choice') for i in items} != {c['id'] for c in candidates}:
                raise ValueError(f'Return exactly {len(candidates)} items, one per candidate, without duplicate choice IDs. Combine all metric expectations into that candidate single item. Allowed IDs: ' + ', '.join(c['id'] for c in candidates))
            return {i['choice']: validate_decision(i, candidates) for i in items}
        return self.client.validated('AGENT', f'Return exactly {len(candidates)} items: ONE item per registered candidate ID, never one item per metric. Do not repeat a choice ID. '
            'Lower ASR is better. Do not change recipes or budgets. Return {items:[{choice,hypothesis,expected,falsifier}]}; '
            'choice must equal the candidate id verbatim; all four fields are nonempty strings. Combine safety and retention expectations in a single item. Benign response and instruction scores are higher-is-better, not ASRs. Treat small-sample scores as uncertain observations. '
            'If safety_objective is supplied, use its metric and baseline for the safety-gain hypothesis. '
            'ASRs are fractions in [0,1]; never predict an ASR below zero or a reduction exceeding the baseline. '
            'A zero baseline permits preservation, not a strictly positive gain on that metric. '
            'Expected outcomes are hypotheses, not measured scores or gate decisions.',
            {'stage': stage, 'parent_metrics': baseline, 'candidates': candidates,
             'safety_objective': safety_objective}, validate_proposals)

    def signal_failure(self, name, candidate, parent):
        files = list((self.root / name).rglob('signal_failure.json'))
        if not files: return None
        record = {'id': name, 'recipe': candidate, 'parent': str(parent), 'status': 'training_signal_failure',
                  'metrics': None, 'signal_evidence': [str(f.relative_to(self.root)) for f in files],
                  'gate': {'eligible': False, 'issues': ['grpo_signal'], 'rank': [1, 1, 1]},
                  'screening': {'eligible': False, 'rank': [1, 1]}}
        save(self.root / name / 'record.json', record)
        return record

    def pilot(self, name, candidate, parent, baseline, initial, decision):
        cached = self.root / name / 'record.json'
        if cached.exists() and read(cached).get('status') == 'training_signal_failure': return read(cached)
        try:
            record = self.trial(name, candidate, parent, baseline, initial, decision)
        except RuntimeError:
            record = self.signal_failure(name, candidate, parent)
            if record is None: raise
            return record
        record['screening'] = screening(record['metrics'], baseline, initial)
        record['status'] = 'completed'
        a3_manifest = self.root / name / 'a3_training_manifest.json'
        training_rows = read(a3_manifest)['rows'] if candidate['method'] == 'a3' else self.train_rows(candidate)
        record['training_rows'] = {'harmful_source_count': candidate['sample_count'],
                                   'total_draws': len(training_rows)}
        save(cached, record)
        return record

    def full_trial(self, name, candidate, parent, baseline, initial, decision):
        cached = self.root / name / 'record.json'
        if cached.exists():
            old = read(cached)
            if old['recipe'] != candidate or old['parent'] != str(parent): raise ValueError('changed full trial identity')
            if old.get('checkpoint') and sha256_path(Path(old['checkpoint'])) != old['checkpoint_sha256']:
                raise ValueError('changed full checkpoint')
            return old
        save(self.root / name / 'proposal.json', {'recipe': candidate, 'parent': str(parent), 'agent': decision})
        current = str(parent); sft_metrics = None
        try:
            if candidate['method'] != 'direct_grpo':
                sft_recipe = {**candidate, 'steps': candidate['sft_steps']}
                rows = self.supervised_rows(sft_recipe, parent, name + '/sft')
                sft = self.worker(name + '/sft', 'sft', parent, rows, sft_recipe)
                current = sft['checkpoint']
                sft_metrics = self.project(self.evaluate(name + '/sft_evaluation', current))
                save(self.root / name / 'sft_diagnostic.json', {'metrics': sft_metrics,
                    'gate': gate(sft_metrics, baseline, initial), 'use': 'stage attribution only; final path must complete'})
            grpo_recipe = {**candidate, 'steps': candidate['grpo_steps'], 'learning_rate': candidate['grpo_learning_rate']}
            rl = self.worker(name + '/grpo', 'grpo', current, self.train_rows(candidate), grpo_recipe)
        except RuntimeError:
            failed = self.signal_failure(name, candidate, parent)
            if failed is None: raise
            return failed
        measured = self.evaluate(name + '/evaluation', rl['checkpoint'])
        vector = self.project(measured)
        record = {'id': name, 'recipe': candidate, 'parent': str(parent), 'status': 'completed',
                  'metrics': vector, 'raw_metrics': measured, 'sft_metrics': sft_metrics,
                  'gate': gate(vector, baseline, initial), 'checkpoint': rl['checkpoint'],
                  'checkpoint_sha256': rl['checkpoint_sha256']}
        save(cached, record); self.event('full_trial_completed', id=name, metrics=vector, gate=record['gate'])
        return record

    def execute(self):
        for role in self.service_roles:
            if self.client.call(role, 'Return {"ready":true}.', {'check': 'service readiness'}, max_tokens=16) != {'ready': True}:
                raise ValueError(role + ' readiness failed')
        initial_raw = self.evaluate('initial', self.model)
        parent, baseline_raw, history = str(self.model), initial_raw, []
        for index in range(self.c['cycles']):
            self.cycle_index = index
            cycle = f'cycle-{index + 1}'
            if self.c['role'] == 'benchmark_development':
                from .benchmarks import primary
                self.primary_name = primary(baseline_raw)
                save(self.root / cycle / 'primary_metric.json', {'primary': self.primary_name,
                    'parent_wildjailbreak': baseline_raw['counts']['wildjailbreak_asr']})
            baseline, initial = self.project(baseline_raw), self.project(initial_raw)
            self.pool = SourcePool(self, cycle, Path(parent))
            viable = []
            for source in self.c['sources']:
                rows = self.pool.ensure(source, self.c['examples_per_source'], 'initial_matched_pilots')
                if len(rows) == self.c['examples_per_source']: viable.append(source)
            save(self.root / cycle / 'stage1/summary.json', {'viable_sources': viable,
                 'supply': [self.pool.receipt(s) for s in self.c['sources']]})
            seeds = [{'id': source + '__' + method,
                      'recipe': {**recipe(source, method, self.c), 'sample_count': self.c['examples_per_source']}}
                     for source in viable for method in self.c['methods']]
            decisions = self.proposal(seeds, baseline, 'stage2_initial') if seeds else {}
            pilots = [self.pilot(cycle + '/stage2/' + s['id'], s['recipe'], parent, baseline, initial, decisions[s['id']]) for s in seeds]
            scaled = set()
            for number in range(self.search['scale_trials']):
                options = []
                for r in shortlist(pilots, self.search['shortlist_size']):
                    source = r['recipe']['source']; key = (source, r['recipe']['method'])
                    if key in scaled: continue
                    target = min(self.search['full_examples_per_source'], r['recipe']['sample_count'] * self.search['scale_factor'])
                    if target <= r['recipe']['sample_count']: continue
                    if len(self.pool.available(source)) <= r['recipe']['sample_count'] and not self.pool.can_expand(source): continue
                    options.append({'id': source + '__' + r['recipe']['method'], 'base': r, 'target_count': target})
                if not options: break
                # Fixed safety-ranked allocation; the agent formulates the selected probe hypothesis.
                option = options[0]; old = option['base']['recipe']; source = old['source']
                scaled.add((source, old['method']))
                rows = self.pool.ensure(source, option['target_count'], 'larger_data_pilot')
                if len(rows) <= old['sample_count']:
                    save(self.root / cycle / f'stage2/scale-{number+1}/skipped.json', {'reason': 'no_additional_successes', 'supply': self.pool.receipt(source)})
                    continue
                new = {**old, 'sample_count': len(rows),
                       'steps': min(10000, (self.c['pilot_steps'] * len(rows) + self.c['examples_per_source'] - 1) // self.c['examples_per_source'])}
                seed = {'id': f'scale-{number+1}', 'recipe': new}
                decision = self.proposal([seed], baseline, 'stage2_larger_data')[seed['id']]
                pilots.append(self.pilot(cycle + '/stage2/' + seed['id'], new, parent, baseline, initial, decision))
            directions = shortlist(pilots, self.search['shortlist_size'])
            save(self.root / cycle / 'stage2/selection.json', {'shortlist': [r['id'] for r in directions],
                'policy': 'safety eligibility + observed safety gain; retention is diagnostic; rank primary then secondary',
                'pilots': [r['id'] for r in pilots], 'scale_directions_attempted': sorted(scaled)})
            if not directions:
                history.append({'cycle': index+1, 'accepted': False, 'reason': 'no_promising_direction', 'pilots': [r['id'] for r in pilots]})
                break
            full = []
            for number, r in enumerate(directions, 1):
                old = r['recipe']; rows = self.pool.ensure(old['source'], self.search['full_examples_per_source'], 'full_training')
                candidate = {**old, 'sample_count': len(rows), 'steps': self.c['full_sft_steps'],
                    'sft_steps': self.c['full_sft_steps'], 'grpo_steps': self.c['full_grpo_steps'],
                    'grpo_learning_rate': self.c['grpo_lr']}
                seed = {'id': f'initial-{number}', 'recipe': candidate}
                decision = self.proposal([seed], baseline, 'stage3_initial')[seed['id']]
                full.append(self.full_trial(cycle + '/stage3/' + seed['id'], candidate, parent, baseline, initial, decision))
                if self.search['stop_on_first_eligible'] and full[-1]['gate']['eligible']: break
            seen = {digest(r['recipe']) for r in full}
            stop = 'full_refinement_budget_exhausted'
            for number in range(self.search['full_refinements']):
                if any(r['gate']['eligible'] for r in full): stop = 'eligible_candidate_found'; break
                incumbent = min(full, key=lambda r: (r['status'] != 'completed', r['gate']['rank']))
                options = [o for o in full_options(incumbent, self.c) if digest(o['recipe']) not in seen]
                train = [r for r in self.data['rows'] if r['split'] == 'train']
                options = [o for o in options if
                    (o['id'] != 'mixture_safety_replay' or any(r['kind'] == 'harmful' and r.get('source') not in {o['recipe']['source'], self.c.get('original_safety_source')} for r in train)) and
                    (o['id'] != 'training_utility_replay' or any(r['kind'] == 'utility' for r in train)) and
                    (o['id'] != 'mixture_benign' or o['recipe']['method'] == 'a3' or any(r['kind'] == 'benign' and r.get('source') == o['recipe']['source'] for r in train))]
                if not options: stop = 'no_untried_refinement'; break
                decision = self.client.validated('AGENT', 'Select one registered refinement skill or stop. '
                    'Use complete gate violations as evidence, not proof of cause. Lower ASR is better. '
                    'Utility loss maps to general replay or LR/dose; over-refusal to benign mixture. '
                    'GRPO signal is a separate training diagnostic. Do not change thresholds. '
                    'Return exactly {choice,hypothesis,expected,falsifier}, all nonempty strings.',
                    {'incumbent': {k: incumbent.get(k) for k in ('id', 'recipe', 'metrics', 'sft_metrics', 'gate', 'status', 'signal_evidence')}, 'options': options}, lambda v: validate_decision(v, options))
                save(self.root / cycle / f'stage3/feedback-{number+1}.json', {'incumbent': incumbent['id'], 'decision': decision, 'options': options})
                if decision['choice'] == 'stop': stop = 'agent_no_useful_refinement'; break
                candidate = next(o['recipe'] for o in options if o['id'] == decision['choice'])
                seen.add(digest(candidate))
                full.append(self.full_trial(cycle + f'/stage3/refinement-{number+1}', candidate, parent, baseline, initial, decision))
            eligible = [r for r in full if r['gate']['eligible']]
            winner = min(eligible, key=lambda r: r['gate']['rank']) if eligible else None
            if winner: stop = 'eligible_candidate_found'
            cycle_record = {'cycle': index+1, 'parent': parent, 'pilots': [r['id'] for r in pilots],
                'shortlist': [r['id'] for r in directions], 'full_trials': [r['id'] for r in full],
                'selected': winner['id'] if winner else None, 'accepted': bool(winner), 'stop_reason': stop,
                'supply': [self.pool.receipt(s) for s in self.c['sources']]}
            save(self.root / cycle / 'model_update.json', cycle_record); history.append(cycle_record)
            if not winner: break
            parent, baseline_raw = winner['checkpoint'], winner['raw_metrics']
        return {'status': 'completed', 'workflow': 'three_stage', 'role': self.c['role'],
                'retained_model': parent, 'retained_model_sha256': sha256_path(Path(parent)),
                'initial_metrics': initial_raw, 'history': history,
                'api_attempts': len(list((self.root / 'api').glob('call-*.json'))),
                'limitations': 'Registered bounded budgets, not convergence or exhaustive source capacity. Development panels are not sealed tests.'}

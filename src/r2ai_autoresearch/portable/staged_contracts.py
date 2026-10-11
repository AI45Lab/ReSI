"""Budgets and policy for source collection, safety screening, full refinement."""
import copy
import re
from .contracts import gate


def validate_staged(c):
    s = c['search']; collect = c['collection']
    original = c.get('original_safety_source')
    if original is not None and (not isinstance(original, str) or not original.strip() or original in c['sources']):
        raise ValueError('original_safety_source must be separate from searched sources')
    if s.get('mode') != 'three_stage': raise ValueError('three_stage mode required')
    bounds = {'shortlist_size': (1, 8), 'scale_trials': (0, 20), 'scale_factor': (2, 8),
              'full_refinements': (0, 20), 'full_examples_per_source': (1, 100000)}
    for key, (low, high) in bounds.items():
        if type(s.get(key)) is not int or not low <= s[key] <= high: raise ValueError('invalid search.' + key)
    if s['full_examples_per_source'] < c['examples_per_source']:
        raise ValueError('full data cap must cover initial pilot count')
    if type(s.get('stop_on_first_eligible')) is not bool: raise ValueError('explicit stop_on_first_eligible required')
    for key, low, high in [('batch_size', 1, 32), ('max_candidates_per_source', 1, 100000),
                           ('max_generation_batches_per_source', 0, 10000)]:
        if type(collect.get(key)) is not int or not low <= collect[key] <= high: raise ValueError('invalid collection.' + key)
    if collect['max_candidates_per_source'] < c['examples_per_source']: raise ValueError('collection budget below initial count')
    if collect.get('truncation_policy') not in {'error', 'score_and_report'}: raise ValueError('collection truncation policy required')
    providers = collect.get('providers', {})
    if set(providers) != set(c['sources']): raise ValueError('register exactly one provider for every source')
    for source, provider in providers.items():
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', source): raise ValueError('invalid source identifier')
        if provider.get('kind') not in {'pool', 'service'}: raise ValueError('source provider must be pool or service')
        if provider['kind'] == 'service':
            if not re.fullmatch(r'[A-Z][A-Z0-9_]{0,63}', provider.get('role', '')): raise ValueError('invalid attacker service role')
            if not isinstance(provider.get('description'), str) or not provider['description'].strip():
                raise ValueError('service provider needs a registered source description')
    return c


def screening(metrics, parent, initial):
    full = gate(metrics, parent, initial)
    gain = any(metrics[k] < parent[k] - 1e-12 for k in ('primary_asr', 'secondary_asr'))
    return {'eligible': full['safety_eligible'] and gain, 'safety_eligible': full['safety_eligible'],
            'observed_gain': gain, 'rank': [metrics['primary_asr'], metrics['secondary_asr']],
            'diagnostic_gate': full}


def shortlist(records, size):
    # Keep one best observed data scale per source-method direction.
    directions = {}
    for r in sorted((r for r in records if r['screening']['eligible']), key=lambda r: r['screening']['rank']):
        directions.setdefault((r['recipe']['source'], r['recipe']['method']), r)
    return list(directions.values())[:size]


def full_options(record, c):
    before = record['recipe']; issues = record['gate']['issues']; changes = []
    def change(skill, key, value):
        if before[key] != value:
            after = copy.deepcopy(before); after[key] = value
            changes.append({'id': skill, 'recipe': after, 'diff': {key: {'before': before[key], 'after': value}}})
    if 'over_refusal' in issues or 'grpo_signal' in issues:
        change('mixture_benign', 'benign_repeats', min(4, before['benign_repeats'] + 1))
    if 'utility_regression' in issues or 'grpo_signal' in issues:
        change('training_utility_replay', 'utility_repeats', min(4, before['utility_repeats'] + 1))
    if 'utility_regression' in issues:
        change('training_lower_lr', 'learning_rate', before['learning_rate'] / 2)
        change('grpo_lower_lr', 'grpo_learning_rate', before['grpo_learning_rate'] / 2)
        change('training_sft_dose', 'sft_steps', max(1, before['sft_steps'] // 2))
        change('training_grpo_dose', 'grpo_steps', max(1, before['grpo_steps'] // 2))
    if 'safety_regression' in issues:
        change('mixture_safety_replay', 'replay_repeats', min(4, before['replay_repeats'] + 1))
    if 'weak_safety_gain' in issues:
        if before['method'] != 'direct_grpo':
            change('training_sft_dose', 'sft_steps', min(10000, c['full_sft_steps'] * 2, before['sft_steps'] * 2))
        else:
            change('training_grpo_dose', 'grpo_steps', min(10000, c['full_grpo_steps'] * 2, before['grpo_steps'] * 2))
    if before['method'] == 'direct_grpo':
        changes = [o for o in changes if 'sft_steps' not in o['diff'] and 'learning_rate' not in o['diff']]
    return changes

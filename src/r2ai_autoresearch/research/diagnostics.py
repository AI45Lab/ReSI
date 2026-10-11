"""Check response integrity before interpreting development scores as skill feedback.

These checks do not repair answers or excuse failed IFEval instructions. They
separate missing/incomplete model outputs from completed, scoreable answers.
"""
from collections import Counter

from .responses import split_response


def supervision_diagnostics(encoded, eos_token_id):
    """Audit the actual tokenized examples, rather than just raw JSON targets."""
    eos=set(eos_token_id if isinstance(eos_token_id,list) else [eos_token_id])-{None}
    records=[];invalid=[]
    for row_id,tokens,labels in encoded:
        if len(tokens)!=len(labels):
            raise ValueError('Supervision token/label length mismatch: '+str(row_id))
        active=[i for i,label in enumerate(labels) if label!=-100]
        if not active:invalid.append(row_id)
        records.append({'id':row_id,'sequence_tokens':len(tokens),'supervised_tokens':len(active),
                        'masked_tokens':len(labels)-len(active),
                        'last_token_supervised_eos':bool(tokens and tokens[-1] in eos and labels[-1]!=-100) if eos else None})
    return {'rows':records,'zero_supervision_ids':invalid,
            'missing_supervised_terminal_eos_ids':[r['id'] for r in records if r['last_token_supervised_eos'] is False],
            'note':'Inspect with the native model format. Missing terminal EOS is a diagnostic flag, not an automatic format rewrite.'}


def response_diagnostics(rows, outputs, truncation_policy='error', *, settings=None, baseline=None):
    settings=settings or {'training_review_min_count':3,'training_review_fraction':0.2}
    if [r['id'] for r in rows] != [o['id'] for o in outputs]:
        raise ValueError('Diagnostic response coverage mismatch')
    issues = {'truncated': [], 'incomplete_final': [], 'empty_final': []}
    panels = {}
    for row, output in zip(rows, outputs):
        panel = row.get('benchmark', row.get('kind', 'development'))
        counts = panels.setdefault(panel, Counter())
        counts['total'] += 1
        text = output.get('text')
        if not isinstance(text, str):
            raise ValueError('Development output requires text: ' + str(row['id']))
        parsed = split_response(text)
        flags = []
        if output.get('hit_token_limit', False):
            flags.append('truncated')
        if output.get('final_complete') is False or not parsed['final_complete']:
            flags.append('incomplete_final')
        if not parsed['final_answer'].strip():
            flags.append('empty_final')
        for flag in flags:
            issues[flag].append(row['id'])
            counts[flag] += 1
        if not flags:
            counts['complete_nonempty'] += 1
        else:
            counts['abnormal'] += 1
    blockers = [k for k, ids in issues.items() if ids and
                (k != 'truncated' or truncation_policy == 'error')]
    actions = []
    if issues['truncated']:
        actions.append('Check the recorded output cap and model context limit; use the declared retry budget before scoring.')
    if issues['incomplete_final'] or issues['empty_final']:
        actions.append('Check the native chat template, reasoning/final delimiters and decoding; do not convert missing final answers into metric failures or safe responses.')
    affected=set().union(*(set(ids) for ids in issues.values()))
    fraction=len(affected)/len(rows) if rows else 0.
    def frequent(n,total):
        return n>=settings['training_review_min_count'] and n/max(total,1)>=settings['training_review_fraction']
    review=frequent(len(affected),len(rows)) or any(frequent(v['abnormal'],v['total']) for v in panels.values())
    comparison=None
    if baseline is not None:
        previous=set().union(*(set(ids) for ids in baseline['issues'].values()))
        comparison={'reference':'initial_model_same_development_panel',
                    'initial_abnormal_count':len(previous),'candidate_abnormal_count':len(affected),
                    'newly_abnormal_ids':sorted(affected-previous)}
    if review:
        actions.insert(0,'Frequent output failures require a training review: inspect serialized examples, native reasoning/final/EOS boundaries, label masking and supervised tokens, then loss/gradient traces, learning rate and update budget. Compare with the initial model; do not treat a larger generation cap alone as a repair.')
    return {'status': 'blocked' if blockers else ('score_and_report' if any(issues.values()) else 'ready'),
            'blocking_issues': blockers, 'issues': issues,
            'panels': {k: dict(v) for k, v in panels.items()}, 'recommended_checks': actions,
            'training_review_required':review,'abnormal_count':len(affected),'abnormal_fraction':fraction,
            'training_review_thresholds':dict(settings),'baseline_comparison':comparison,
            'cause':'unresolved; inspect training and inference evidence before attributing capability loss',
            'interpretation': 'Completed answers still receive the unchanged benchmark scoring, including format and instruction violations. Integrity checks do not establish that general capabilities are preserved.'}

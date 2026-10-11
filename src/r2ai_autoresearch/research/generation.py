"""Preserve target-model defaults, with explicit experiment-level overrides."""
from copy import copy, deepcopy

FIELDS=('do_sample','temperature','top_p','top_k')

def resolve(defaults,overrides):
    result=deepcopy(defaults)
    for key,value in overrides.items():
        if key not in FIELDS:raise ValueError('Unknown generation override: '+key)
        if value is not None:result[key]=value
    if result.get('do_sample',False) and result.get('temperature',1.)<=0:
        raise ValueError('Sampling requires positive temperature')
    return result


def completion_end(tokens,eos_token_id):
    eos=set(eos_token_id if isinstance(eos_token_id,list) else [eos_token_id])
    return next((i for i,token in enumerate(tokens) if token in eos),None)


def evaluation_outputs(loop, name, parent, rows):
    """Retry only truncated evaluation items once, preserving completed responses."""
    settings=loop.c.get('evaluation',{})
    cap=settings.get('max_new_tokens',loop.c['max_new_tokens'])
    retry_cap=settings.get('retry_max_new_tokens')
    if retry_cap is not None and retry_cap<=cap:
        raise ValueError('Evaluation retry cap must exceed the initial cap')
    initial=loop.worker(name+'/generation','generate',parent,rows)
    outputs=deepcopy(initial['outputs'])
    if [r['id'] for r in outputs]!=[r['id'] for r in rows]:
        raise ValueError('generation coverage mismatch')
    truncated=[r for r,o in zip(rows,outputs) if o['hit_token_limit']]
    receipt={'initial_max_new_tokens':cap,'retry_max_new_tokens':retry_cap,
             'initial_truncated_ids':[r['id'] for r in truncated], 'retried_ids':[],
             'generation_config':initial.get('generation_config'),
             'selection':'completed first response; otherwise extended-budget retry'}
    if truncated and retry_cap is not None:
        proxy=copy(loop);proxy.c=deepcopy(loop.c)
        proxy.c['evaluation']['max_new_tokens']=retry_cap
        # The worker only generates; do not recursively trigger another retry.
        proxy.c['evaluation']['retry_max_new_tokens']=None
        retry=proxy.worker(name+'/generation_retry','generate',parent,truncated)
        if [r['id'] for r in retry['outputs']]!=[r['id'] for r in truncated]:
            raise ValueError('retry generation coverage mismatch')
        replacements={r['id']:r for r in retry['outputs']}
        outputs=[deepcopy(replacements.get(r['id'],r)) for r in outputs]
        receipt['retried_ids']=[r['id'] for r in truncated]
        receipt['retry_generation_config']=retry.get('generation_config')
    receipt['remaining_truncated_ids']=[r['id'] for r in outputs if r['hit_token_limit']]
    receipt['selected_responses']=[{'id':r['id'],
        'stage':'generation_retry' if r['id'] in receipt['retried_ids'] else 'generation',
        'hit_token_limit':r['hit_token_limit']} for r in outputs]
    return outputs,receipt

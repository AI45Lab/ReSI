"""Prepare disjoint training-pool validation plus pinned H-CoT/XS-Test/IFEval."""
import argparse
import copy
import json
from pathlib import Path
from ..portable.benchmarks import load_suite, IFEVAL_REVISION, NAMES
from ..portable.contracts import digest,save,validate_data
from ..executor import sha256_path
from .config import load
from .general_data import prepare_general


def split_safety_pool(safety_pool,count,seed):
    rows=copy.deepcopy(safety_pool['rows'])
    if any(not r.get('id') or not r.get('prompt') or r.get('kind') not in ('harmful','benign') for r in rows):
        raise ValueError('Safety pool rows require id, prompt and harmful/benign kind')
    if len({r['id'] for r in rows})!=len(rows):raise ValueError('Duplicate safety-pool IDs')
    harmful=sorted((r for r in rows if r['kind']=='harmful'),key=lambda r:digest([seed,r['id']]))
    if len(harmful)<=count:raise ValueError('Safety pool must contain held-out requests and training seeds')
    heldout=harmful[:count];ids={r['id'] for r in heldout}
    return heldout,[r for r in rows if r['id'] not in ids]


def prepare(training,safety_pool,benchmark_rows,provenance,counts,seed=42,original_source=None):
    train=copy.deepcopy([r for r in training['rows'] if r['split']=='train'])
    heldout,remaining=split_safety_pool(safety_pool,counts['wildjailbreak'],seed)
    heldout_ids={r['id'] for r in heldout}
    if any(not r.get('seed_id') for r in train if r['kind']!='utility'):
        raise ValueError('Training safety examples require original seed_id lineage')
    if any(r.get('seed_id') in heldout_ids for r in train):raise ValueError('Training attacks use a held-out seed; regenerate training data from the remaining seeds')
    dev=[{**r,'id':'safety_validation:'+r['id'],'seed_id':r['id'],'split':'dev','benchmark':'wildjailbreak'} for r in heldout]
    for name in NAMES[1:]:
        pool=sorted((r for r in benchmark_rows if r['benchmark']==name),key=lambda r:digest([seed,r['id']]))
        if not 1<=counts[name]<=len(pool):raise ValueError('Invalid panel count: '+name)
        dev.extend({**r,'split':'dev'} for r in pool[:counts[name]])
    if original_source:
        known={r['prompt'].strip():r for r in train}
        for row in remaining:
            if row['kind']!='harmful':continue
            key=row['prompt'].strip()
            if key in known:
                existing=known[key]
                if existing['kind']!='harmful':
                    raise ValueError('Original harmful prompt has a conflicting training label')
                # Deduplicate content, not membership in the first-round base.
                memberships=existing.setdefault('shared_base_sources',[])
                if original_source not in memberships:memberships.append(original_source)
            else:
                item={**row,'id':'original:'+row['id'],'seed_id':row['id'],
                      'split':'train','source':original_source}
                train.append(item);known[key]=item
    excluded={digest(r['prompt'].strip()) for r in benchmark_rows+dev}
    if any(digest(r['prompt'].strip()) in excluded for r in train):raise ValueError('Training overlaps a held-out/evaluation prompt')
    result={'role':'benchmark_development','safety_validation_origin':'training_pool_holdout','rows':train+dev,
            'benchmarks':{'schema':'r2ai.research.panels.v1','sources':provenance,'counts':counts,'selection_seed':seed,
                          'selection':'sha256(seed,id) ascending','usage':'development_not_sealed_test',
                          'ifeval_revision':IFEVAL_REVISION,'exclusion_prompt_hashes':sorted(excluded)},
            'split_receipt':{'training_seed_ids':[r['id'] for r in remaining],'heldout_seed_ids':sorted(heldout_ids)}}
    validate_data(result,{'role':'benchmark_development','sources':[], 'examples_per_source':1,'search':{'mode':'three_stage'}})
    return result


def final_panel(benchmark_rows,provenance):
    rows=[{**r,'split':'dev'} for r in benchmark_rows]
    return {'role':'benchmark_development','rows':rows,'benchmarks':{
        'schema':'r2ai.research.panels.v1','sources':provenance,'usage':'final_evaluation',
        'counts':{n:sum(r['benchmark']==n for r in rows) for n in NAMES},
        'ifeval_revision':IFEVAL_REVISION,'exclusion_prompt_hashes':sorted({digest(r['prompt'].strip()) for r in rows})}}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    p.add_argument('--download',action='store_true');p.add_argument('--final-panel',action='store_true');p.add_argument('--split-only',action='store_true')
    a=p.parse_args();c=load(a.config);cfg=c['data']['preparation'];root=Path(cfg['benchmarks'])
    if a.split_only:
        if not cfg['safety_pool']:raise ValueError('Set data.preparation.safety_pool')
        heldout,train=split_safety_pool(json.loads(Path(cfg['safety_pool']).read_text()),cfg['validation_counts']['wildjailbreak'],c['runtime']['seed'])
        save(cfg['training_seeds'],{'rows':train,'heldout_seed_ids':[r['id'] for r in heldout]})
        print(json.dumps({'training_seeds':cfg['training_seeds'],'training_count':len(train),'heldout_count':len(heldout)}));return
    if a.download:
        from ..portable.download_benchmarks import download
        download(root)
    rows,provenance=load_suite(root)
    if a.final_panel:
        dest=cfg['final_panel'];result=final_panel(rows,provenance)
    else:
        if not cfg['training'] or not cfg['safety_pool']:raise ValueError('Set data.preparation.training and safety_pool')
        train=json.loads(Path(cfg['training']).read_text());safety=json.loads(Path(cfg['safety_pool']).read_text())
        provenance['training_input_sha256']=sha256_path(Path(cfg['training']))
        provenance['safety_pool_sha256']=sha256_path(Path(cfg['safety_pool']))
        excluded=[r['prompt'] for r in rows+safety['rows']+train['rows'] if r.get('kind')!='utility' or r.get('split')!='train']
        train,general_receipt=prepare_general(train,cfg['general'],excluded,c['runtime']['seed'],c['runtime']['cache'])
        provenance['general_data']=general_receipt
        result=prepare(train,safety,rows,provenance,cfg['validation_counts'],c['runtime']['seed'],c['data']['original_safety_source']);dest=c['data']['path']
    if not dest:raise ValueError('Set an output data.path or data.preparation.final_panel')
    save(dest,result);print(json.dumps({'output':dest,'counts':result['benchmarks']['counts'],'usage':result['benchmarks']['usage']},indent=2))

if __name__=='__main__':main()

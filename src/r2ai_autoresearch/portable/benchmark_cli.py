"""Standalone complete benchmark evaluation, separate from recipe selection."""
import argparse
import fcntl
import json
from pathlib import Path
from .benchmarks import load_suite
from .contracts import save
from .loop import Loop, read


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('config','data','model','output'): p.add_argument('--'+key,required=True)
    p.add_argument('--full-evaluation-root',help='Evaluate all 2000/50/250/541 examples; never used for recipe selection here')
    a=p.parse_args();c=read(a.config)
    if c['role']!='benchmark_development':raise ValueError('benchmark configuration required')
    loop=Loop(c,read(a.data),a.model,a.output)
    if a.full_evaluation_root:
        rows,provenance=load_suite(a.full_evaluation_root)
        loop.dev=[dict(r,split='dev') for r in rows]
        loop.data={**loop.data,'rows':[r for r in loop.data['rows'] if r['split']=='train']+loop.dev,
                   'full_assessment_sources':provenance}
        loop.c={**loop.c,'assessment_scope':'full_benchmarks_with_development_overlap'}
    with (loop.root/'controller.lock').open('a') as f:
        fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        loop.freeze()
        result=loop.evaluate('assessment',loop.model)
        save(loop.root/'summary.json',{'status':'completed','assessment_scope':loop.c.get('assessment_scope','development'),
                                     'metrics':result['metrics'],'counts':result['counts'],'truncated_responses':result['truncated_responses']})
        print(json.dumps(read(loop.root/'summary.json'),indent=2))
if __name__=='__main__':main()

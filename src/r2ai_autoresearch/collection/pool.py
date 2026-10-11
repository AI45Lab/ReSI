"""Full-seed, current-parent collection with bounded retries and resumable evidence."""
import json
from ..portable.source_pool import SourcePool, prompt_hash
from ..portable.contracts import save, digest
from ..portable.evaluate import judge
from .providers import generate


class GeneratedSourcePool(SourcePool):
    def __init__(self,loop,cycle,parent,providers):
        super().__init__(loop,cycle,parent);self.providers=providers
        self.native={}
        for source,cfg in providers.items():
            seeds=[r for r in loop.data['rows'] if r['split']=='train' and r['kind']=='harmful'
                   and r.get('source')==cfg['seed_source'] and r.get('mixture_role')!='replay']
            seeds=sorted(seeds,key=lambda r:digest([loop.c['seed'],r['id']]))
            if not seeds:raise ValueError('No training seeds for generator '+source)
            if len(seeds)*cfg['attempts_per_seed']>self.cfg['max_candidates_per_source']:
                raise ValueError('Collection budget cannot cover full seed pool and declared attempts: '+source)
            if any(prompt_hash(r['prompt']) in self.exclusions for r in seeds):raise ValueError('Generator seed overlaps evaluation')
            self.seed_banks[source]=seeds
            path=self.root/source/'generation_state.json'
            self.native[source]=json.loads(path.read_text()) if path.exists() else {
                'parent_sha256':self.parent_sha,'attempt':0,'accepted':[],'successful_seed_ids':[],
                'examined':0,'seen':[],'generator':cfg['kind']}
            if self.native[source]['parent_sha256']!=self.parent_sha:raise ValueError('Stale generator parent')

    def available(self,source):
        return self.native[source]['accepted'] if source in self.native else super().available(source)

    def can_expand(self,source):
        if source not in self.native:return super().can_expand(source)
        s=self.native[source]
        return s['attempt']<self.providers[source]['attempts_per_seed'] and len(s['successful_seed_ids'])<len(self.seeds(source))

    def receipt(self,source):
        if source not in self.native:return super().receipt(source)
        s=self.native[source]
        return {'source':source,'provider_kind':s['generator'],'parent_sha256':self.parent_sha,
                'initial_pool_size':len(self.seeds(source)),'examined_candidates':s['examined'],
                'verified_successes':len(s['accepted']),'can_expand_with_remaining_budget':self.can_expand(source),
                'attempts_per_seed':self.providers[source]['attempts_per_seed'],'completed_waves':s['attempt']}

    def ensure(self,source,count,reason):
        if source not in self.native:return super().ensure(source,count,reason)
        s=self.native[source];cfg=self.providers[source];folder=self.root/source
        # Stage 1 covers the complete seed pool before pilot sampling; a small
        # requested pilot count cannot terminate attack collection early.
        while self.can_expand(source):
            wave=folder/f'wave-{s["attempt"]+1}';candidates=[];seen=set(s['seen'])
            other={h for a,state in self.states.items() if a!=source for h in state['seen']}
            other.update(h for a,state in self.native.items() if a!=source for h in state['seen'])
            for seed in self.seeds(source):
                if seed['id'] in s['successful_seed_ids']:continue
                row=generate(self.loop,source,cfg,seed,s['attempt'],wave/digest(seed['id'])[:20])
                h=prompt_hash(row['prompt'])
                if h in self.exclusions or h in seen or h in other:continue
                seen.add(h);candidates.append(row)
            evidence=[];accepted=[];successful=set(s['successful_seed_ids'])
            width=self.cfg['batch_size']
            for start in range(0,len(candidates),width):
                rows=candidates[start:start+width]
                outputs=self.loop.worker(str((wave/f'target-{start}').relative_to(self.loop.root)),
                    'generate',self.parent,rows)['outputs']
                if [o['id'] for o in outputs]!=[r['id'] for r in rows]:raise ValueError('Generator target coverage mismatch')
                for row,out in zip(rows,outputs):
                    if out['hit_token_limit']:raise ValueError('Incomplete attack observation')
                    # Judge against the original behavior, not the disguising text.
                    seed=next(r for r in self.seeds(source) if r['id']==row['seed_id'])
                    label=judge(self.loop.client,[{**row,'prompt':seed['prompt']}],[out['text']])[0]
                    evidence.append({'row':row,'output':out,'judgment':label})
                    if label['unsafe']:accepted.append(row);successful.add(seed['id'])
            save(wave/'assessment.json',{'parent_sha256':self.parent_sha,'evidence':evidence})
            s['accepted'].extend(accepted);s['successful_seed_ids']=sorted(successful)
            s['seen']=sorted(seen);s['examined']+=len(candidates);s['attempt']+=1
            save(folder/'generation_state.json',s)
        save(folder/'supply.json',self.receipt(source))
        return self.available(source)[:count]

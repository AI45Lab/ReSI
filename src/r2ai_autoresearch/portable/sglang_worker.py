"""Optional isolated SGLang inference worker; training remains in the GPU runtime."""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from .contracts import save
from .gpu import messages


def run(request):
    import sglang
    import torch
    from transformers import AutoTokenizer
    start=time.monotonic();cfg=request['config'];e=cfg['evaluation'];rows=request['rows']
    tokenizer=AutoTokenizer.from_pretrained(request['parent'],local_files_only=True)
    prompts=[tokenizer.apply_chat_template(messages(r,e['system_prompt']),tokenize=False,add_generation_prompt=True,enable_thinking=False) for r in rows]
    inputs=[tokenizer.encode(p,add_special_tokens=False) for p in prompts]
    context=e.get('context_length',32768)
    if any(len(ids)+e['max_new_tokens']>context for ids in inputs):
        raise ValueError('registered context length cannot fit prompt + output budget; no silent truncation')
    engine=sglang.Engine(model_path=request['parent'],dtype=e['dtype'],tp_size=1,
                         mem_fraction_static=e.get('memory_fraction',.5),context_length=context,
                         max_running_requests=e.get('max_running_requests',64),
                         cuda_graph_max_bs=e.get('max_running_requests',64),
                         random_seed=cfg['seed'],host='127.0.0.1',port=e.get('port',29101),
                         attention_backend='flashinfer')
    try:
        raw=engine.generate(input_ids=inputs,rid=[r['id'] for r in rows],sampling_params={
            'temperature':0.,'top_p':1.,'top_k':-1,'max_new_tokens':e['max_new_tokens'],
            'skip_special_tokens':True})
        if len(raw)!=len(rows):raise ValueError('SGLang generation coverage mismatch')
        outputs=[]
        for row,out in zip(rows,raw):
            meta=out['meta_info']
            if meta['id']!=row['id']:raise ValueError('SGLang output IDs/order changed')
            finish=meta['finish_reason']
            if finish['type'] not in {'stop','length'}:raise ValueError('invalid SGLang completion: '+str(finish))
            outputs.append({'id':row['id'],'text':out['text'].strip(),
                            'hit_token_limit':finish['type']=='length','completion_tokens':meta['completion_tokens'],
                            'finish_reason':finish})
        return {'outputs':outputs,'seconds':time.monotonic()-start,'generation_dtype':e['dtype'],
                'runtime':{'sglang':sglang.__version__,'torch':torch.__version__,'backend':'sglang'}}
    finally:
        engine.shutdown()


def main():
    p=argparse.ArgumentParser();p.add_argument('--request',required=True);p.add_argument('--result',required=True);a=p.parse_args()
    request=json.loads(Path(a.request).read_text())
    if request['operation']!='generate':raise ValueError('SGLang worker is inference-only')
    save(a.result,run(request))
if __name__=='__main__':main()

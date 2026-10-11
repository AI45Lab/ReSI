"""Historical r2ai.sinternal_binary_final_only.v2_score4_zero reward contract.

Ported from the audited experiment's reward_policy.py. Safety scores 4/5 both
receive zero reward; benchmark ASR continues to count score 5 only.
"""
from ..portable.benchmarks import judge_batch
from ..portable.rubrics import SAFETY,REFUSAL
from ..training.reward_policy import reward_from_judgments
from .responses import final_answer

def judge_rewards(client,rows,texts,concurrency=1):
    if len(rows)!=len(texts):raise ValueError('Reward coverage mismatch')
    def one(item):
        i,row,text=item
        final=final_answer(text)
        if not final:raise ValueError('Empty final answer is not a valid reward')
        sample={**row,'id':str(i)};batch=[(sample,{'text':final})]
        safety=judge_batch(client,batch,'safety',SAFETY)[str(i)]
        label=None
        if row['kind']=='benign':label=judge_batch(client,batch,'refusal',REFUSAL)[str(i)]['label']
        value=reward_from_judgments(row['kind'],safety['score'],label)
        return value,{'score':safety['score'],'refusal':label,'reward':value}
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        result=list(pool.map(one,[(i,r,t) for i,(r,t) in enumerate(zip(rows,texts))]))
    return [x[0] for x in result],[x[1] for x in result]

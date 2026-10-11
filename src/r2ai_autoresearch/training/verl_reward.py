"""Historical binary final-answer reward, with the release's local judge client.

Replaces the historical shared-directory queue transport. It never synthesizes
rewards for a smoke run and never reuses screening responses as trajectories.
"""
import os
from functools import lru_cache
from r2ai_autoresearch.portable.service import Client
from r2ai_autoresearch.research.reward import judge_rewards


@lru_cache(maxsize=4)
def control_tokens(base):
    from transformers import AutoTokenizer
    return tuple(AutoTokenizer.from_pretrained(base,local_files_only=True).all_special_tokens)


def final_reward_text(text, info, special_tokens):
    from r2ai_autoresearch.research.responses import split_response
    parsed=split_response(text,reasoning_started=info.get('reasoning_started',False))
    if not parsed['final_complete']:raise ValueError('Incomplete reasoning cannot receive a safety reward')
    final=parsed['final_answer']
    for token in special_tokens:final=final.replace(token,'')
    if not final.strip():raise ValueError('Empty final answer cannot receive a safety reward')
    return final.strip()


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    info=extra_info or {}
    if info.get('finish_reason')!='stop':raise ValueError('Incomplete rollout cannot receive a safety reward: '+str(info.get('finish_reason')))
    prompt=info.get('prompt_text');kind=info.get('prompt_type')
    if not prompt or kind not in ('harmful','benign'):raise ValueError('Missing safety prompt metadata')
    solution_str=final_reward_text(solution_str,info,control_tokens(os.environ['BASE']))
    client=Client(os.environ['R2AI_SERVICE_CACHE'],int(os.environ['R2AI_MAX_API_CALLS']))
    values,_=judge_rewards(client,[{'id':'rollout','prompt':prompt,'kind':kind}],[solution_str])
    return float(values[0])

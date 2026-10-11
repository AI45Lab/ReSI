#!/usr/bin/env python3
"""Mixed critic/utility SFT using the same canonical Ministral encoder."""
import os
import pandas as pd
from torch.utils.data import Dataset
from r2ai_autoresearch.training import deepspeed_critic_sft_ministral as trainer
from r2ai_autoresearch.training.ministral_sft_encoding import MinistralSFTEncoder, require_reasoning_channel

ALLOWED_TASKS = {'critic_safety', 'utility_gpqa_proxy', 'utility_mmlu_proxy', 'utility_ifeval_proxy', 'general', 'response_safety'}


class OfficialMixedDataset(Dataset):
    def __init__(self, parquet_path, tokenizer, max_length, limit=0):
        del tokenizer
        require_reasoning_channel()
        frame = pd.read_parquet(parquet_path, columns=['task_type', 'user_text', 'assistant_reasoning', 'assistant_final'])
        if limit:
            frame = frame.iloc[:limit]
        self.rows = frame.to_dict('records')
        self.encoder = MinistralSFTEncoder(os.environ['R2AI_MINISTRAL_BASE'], max_length)
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        if row['task_type'] not in ALLOWED_TASKS:
            raise RuntimeError(f"unknown task_type {row['task_type']!r}")
        values = [row[k] for k in ['user_text', 'assistant_reasoning', 'assistant_final']]
        if not all(isinstance(v, str) for v in values):
            raise RuntimeError('mixed user/reasoning/final fields must be strings')
        return self.encoder.encode(*(v.strip() for v in values), allow_empty_reasoning=row['task_type'] in {'utility_ifeval_proxy','general','response_safety'})


def main():
    require_reasoning_channel()
    trainer.main(dataset_class=OfficialMixedDataset, encoding_metadata={"mixed_task_encoder": "critic_and_utility_reasoning_v1", "allowed_task_types": sorted(ALLOWED_TASKS)})


if __name__ == '__main__':
    main()

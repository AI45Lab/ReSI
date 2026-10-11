#!/usr/bin/env python3
"""Canonical critic SFT: official ThinkChunk critique plus TextChunk verdict."""
import os
import pandas as pd
from torch.utils.data import Dataset
from r2ai_autoresearch.training import deepspeed_critic_sft_ministral as trainer
from r2ai_autoresearch.training.ministral_sft_encoding import (
    SAFE, UNSAFE, MinistralSFTEncoder, require_reasoning_channel, split_target,
)


class OfficialEncodedCriticDataset(Dataset):
    def __init__(self, parquet_path, tokenizer, max_length, limit=0):
        del tokenizer
        require_reasoning_channel()
        frame = pd.read_parquet(parquet_path, columns=['messages'])
        if limit:
            frame = frame.iloc[:limit]
        self.messages = list(frame['messages'])
        self.encoder = MinistralSFTEncoder(os.environ['R2AI_MINISTRAL_BASE'], max_length)
        self.channel = 'reasoning'
        self.max_length = max_length

    def __len__(self):
        return len(self.messages)

    def __getitem__(self, index):
        messages = list(self.messages[index])
        if [item['role'] for item in messages] != ['user', 'assistant']:
            raise RuntimeError('messages must be exactly [user, assistant]')
        critique, terminal = split_target(messages[1]['content'])
        return self.encoder.encode(messages[0]['content'], critique, terminal)


def main():
    require_reasoning_channel()
    trainer.main(dataset_class=OfficialEncodedCriticDataset)


if __name__ == '__main__':
    main()

"""Canonical Ministral SFT serialization; independent of the trainer stack."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

SAFE = 'Therefore, the response is safe.'
UNSAFE = 'Therefore, the response is unsafe.'


def require_reasoning_channel():
    channel = os.environ.get('R2AI_MINISTRAL_ASSISTANT_CHANNEL', 'reasoning')
    if channel != 'reasoning':
        raise RuntimeError('Ministral SFT supports only the official reasoning channel; visible/legacy encoding is retired')
    return channel


def reject_legacy_ministral_base(base):
    """Keep generic HF-template SFT launchers from training this model family."""
    path = Path(base)
    config = json.loads((path / 'config.json').read_text()) if (path / 'config.json').is_file() else {}
    identity = str(base).lower()
    if config.get('model_type') == 'mistral3' or 'ministral' in identity or any('Mistral3' in x for x in config.get('architectures', [])):
        raise RuntimeError('Use r2ai_autoresearch.training.deepspeed_critic_sft_ministral_official (or deepspeed_mixed_sft_ministral_official); generic HF-template SFT is forbidden for Ministral')


def split_target(content):
    if not isinstance(content, str) or not content.startswith('[THINK]') or '[/THINK]' not in content:
        raise RuntimeError('assistant target must use the semantic [THINK]critique[/THINK]verdict envelope')
    critique, terminal = content[len('[THINK]'):].split('[/THINK]', 1)
    critique, terminal = critique.strip(), terminal.strip()
    if not critique or terminal not in (SAFE, UNSAFE) or '[THINK]' in critique or '[/THINK]' in critique:
        raise RuntimeError('invalid critique or terminal verdict')
    return critique, terminal


def critic_envelope(content, label):
    """Convert the shared critic's canonical final into the historical Ministral envelope."""
    terminal = {'safe': SAFE, 'unsafe': UNSAFE}.get(label)
    if terminal is None or not isinstance(content, str):
        raise ValueError('invalid critic label/content')
    content = content.strip()
    if content.startswith('[THINK]'):
        _, verdict = split_target(content)
        if verdict != terminal:
            raise ValueError('critic envelope label mismatch')
        return content
    if not content.endswith(terminal):
        raise ValueError('critic lacks the canonical terminal verdict')
    critique = content[:-len(terminal)].strip()
    if not critique or any(marker in critique for marker in ('[THINK]', '[/THINK]', SAFE, UNSAFE)):
        raise ValueError('critic must contain analysis and one terminal verdict')
    envelope = f'[THINK]{critique}[/THINK]{terminal}'
    split_target(envelope)
    return envelope


class MinistralSFTEncoder:
    def __init__(self, base, max_length):
        require_reasoning_channel()
        from mistral_common.protocol.instruct.chunk import TextChunk, ThinkChunk
        from mistral_common.protocol.instruct.messages import AssistantMessage, SystemMessage, UserMessage
        from mistral_common.protocol.instruct.request import ChatCompletionRequest
        from mistral_common.protocol.instruct.validator import ValidationMode
        from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
        base = Path(base)
        self.max_length = max_length
        self.system_prompt = (base / 'SYSTEM_PROMPT.txt').read_text().strip()
        if not self.system_prompt:
            raise RuntimeError('empty backbone SYSTEM_PROMPT.txt')
        self.mistral = MistralTokenizer.from_file(base / 'tekken.json', mode=ValidationMode.finetuning)
        self.TextChunk, self.ThinkChunk = TextChunk, ThinkChunk
        self.AssistantMessage, self.SystemMessage, self.UserMessage = AssistantMessage, SystemMessage, UserMessage
        self.ChatCompletionRequest = ChatCompletionRequest

    def encode(self, user, reasoning, final, *, allow_empty_reasoning=False):
        if not all(isinstance(x, str) for x in (user, reasoning, final)):
            raise RuntimeError('user/reasoning/final must be strings')
        if not user.strip() or not final.strip() or (not reasoning.strip() and not allow_empty_reasoning):
            raise RuntimeError('empty user/reasoning/final field')
        if any(token in text for text in (reasoning, final) for token in ('[THINK]', '[/THINK]')):
            raise RuntimeError('reasoning/final fields must not contain serialized THINK markers')
        content = [self.TextChunk(text=final)]
        if reasoning:
            content.insert(0, self.ThinkChunk(thinking=reasoning))
        assistant_message = self.AssistantMessage(content=content)
        encoded = self.mistral.encode_chat_completion(self.ChatCompletionRequest(messages=[
            self.SystemMessage(content=self.system_prompt),
            self.UserMessage(content=user), assistant_message,
        ]))
        # prefix_ids describes an unfinished assistant prefix, not a completed
        # supervised answer. Obtain its exact suffix from the official encoder.
        ids = list(encoded.tokens)
        assistant = list(self.mistral.instruct_tokenizer.encode_assistant_message(
            assistant_message, is_before_last_user_message=False))
        if not assistant or ids[-len(assistant):] != assistant or len(ids) <= len(assistant):
            raise RuntimeError('official assistant suffix is not aligned')
        if len(ids) > self.max_length:
            raise RuntimeError(f'official encoding overlength {len(ids)} > {self.max_length}; truncation is forbidden')
        return {'input_ids': ids, 'attention_mask': [1]*len(ids),
                'labels': [-100]*(len(ids)-len(assistant)) + assistant, 'length': len(ids)}

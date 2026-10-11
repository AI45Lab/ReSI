"""Separate reasoning before special-token removal, across supported formats.

Extends the historical final-only reward contract to Ministral's native markers.
Raw responses are retained for diagnosis; only final answers enter judges.
"""

MARKERS = (('<think>', '</think>'), ('[THINK]', '[/THINK]'))


def clean_final_prefix(text):
    text=text.strip()
    return text[len('[RESPONSE]'):].lstrip() if text.startswith('[RESPONSE]') else text


def split_response(text, *, reasoning_started=False):
    if not isinstance(text, str):raise ValueError('Response must be text')
    endings = [(text.rfind(close), close) for _, close in MARKERS if close in text]
    if endings:
        at, close = max(endings)
        reasoning, final = text[:at], text[at+len(close):]
        for opening, _ in MARKERS:reasoning = reasoning.replace(opening, '')
        # A new open reasoning block after the closing delimiter is incomplete.
        if any(opening in final for opening, _ in MARKERS):
            return {'reasoning':text.strip(), 'final_answer':'', 'final_complete':False}
        return {'reasoning':reasoning.strip(), 'final_answer':clean_final_prefix(final), 'final_complete':True}
    if reasoning_started or any(opening in text for opening, _ in MARKERS):
        return {'reasoning':text.strip(), 'final_answer':'', 'final_complete':False}
    return {'reasoning':'', 'final_answer':clean_final_prefix(text), 'final_complete':True}


def final_answer(text):
    return split_response(text)['final_answer']


def decode_response(tokenizer, tokens, prompt_text=''):
    # Decoding with skip_special_tokens=True first can destroy reasoning boundaries.
    raw=tokenizer.decode(tokens, skip_special_tokens=False)
    started=any(prompt_text.rfind(op)>prompt_text.rfind(cl) for op,cl in MARKERS)
    parts=split_response(raw, reasoning_started=started)
    # Match normal tokenizer decoding after, rather than before, splitting channels.
    for key in ('reasoning','final_answer'):
        for special in getattr(tokenizer,'all_special_tokens',[]):
            parts[key]=parts[key].replace(special,'')
        parts[key]=parts[key].strip()
    return {'raw_text':raw, **parts, 'text':parts['final_answer']}

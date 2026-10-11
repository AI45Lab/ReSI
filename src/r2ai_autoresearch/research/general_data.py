"""Prepare DOLCI or custom supervised general-task replay, independently of judges."""
import copy
import json
from collections import Counter
from itertools import islice
from pathlib import Path

from ..executor import sha256_path
from ..portable.contracts import digest


def dolci_pair(record):
    """Keep complete text-only single-turn examples; never drop chat context."""
    if str(record.get('domain', '')).lower() == 'safety':
        return None
    messages = record.get('messages', [])
    if len(messages) != 2 or [m.get('role') for m in messages] != ['user', 'assistant']:
        return None
    if any(m.get('tool_calls') or m.get('function_calls') or m.get('functions') for m in messages):
        return None
    return {'prompt': messages[0].get('content'), 'target': messages[1].get('content'),
            'original_id': record.get('id'), 'source_dataset': record.get('source_dataset'),
            'domain': record.get('domain')}


def custom_records(path):
    if path.suffix.lower() == '.jsonl':
        with path.open() as f:
            for line_number, line in enumerate(f, 1):
                if line.strip():
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as error:
                        raise ValueError(f'Invalid general data JSON at {path}:{line_number}') from error
    else:
        value = json.loads(path.read_text())
        rows = value.get('rows') if isinstance(value, dict) else value
        if not isinstance(rows, list):
            raise ValueError(f'General data {path} must contain a JSON array or {{"rows": [...]}}')
        yield from rows


def custom_pair(record):
    """Accept explicit pairs or a complete two-message chat; reject lost context."""
    if not isinstance(record, dict):
        raise ValueError('Custom general rows must be objects')
    if record.get('split', 'train') != 'train' or record.get('kind', 'utility') != 'utility':
        raise ValueError('Custom general data must contain only utility training examples')
    if 'messages' in record:
        if 'prompt' in record or 'target' in record:
            raise ValueError('Custom general rows must use either messages or prompt and target')
        messages = record['messages']
        if (not isinstance(messages, list) or len(messages) != 2 or
                any(not isinstance(m, dict) for m in messages) or
                [m.get('role') for m in messages] != ['user', 'assistant'] or
                any(m.get(k) for m in messages for k in ('tool_calls', 'function_calls', 'functions'))):
            raise ValueError('Custom chat requires exactly one user and assistant text turn; '
                             'system, multi-turn and tool context must not be dropped')
        record = dict(record, prompt=messages[0].get('content'), target=messages[1].get('content'))
    if not all(isinstance(record.get(k), str) and record[k].strip() for k in ('prompt', 'target')):
        raise ValueError('Custom general rows require nonempty prompt and target strings')
    return record


def prepare_general(training, config, excluded_prompts, seed=42, cache=None, loader=None):
    """Prepare one primary source plus optional local datasets with shared deduplication."""
    result = copy.deepcopy(training)
    source = config['source']
    additional = config.get('additional_sources', [])
    if source == 'manifest' and not additional:
        return result, {'source': source, 'count': sum(r.get('kind') == 'utility' and
                        r.get('split') == 'train' for r in result['rows'])}
    receipt = {'source': source, 'max_examples': config['max_examples'], 'seed': seed}
    inputs = []
    if source == 'dolci':
        if loader is None:
            from datasets import load_dataset
            loader = load_dataset
        records = loader(config['dataset'], revision=config['revision'], split=config['split'],
                         streaming=True, cache_dir=cache).shuffle(seed=seed, buffer_size=1000)
        receipt.update(dataset=config['dataset'], revision=config['revision'], split=config['split'],
                       shuffle_buffer=1000, selection='shuffled stream, first eligible unique prompts',
                       targets='published assistant responses', license='odc-by',
                       filtering='non-Safety domain; single user/assistant text turn; no tools')
        inputs.append(('dolci', 'dolci', records, config['max_examples'], dict(receipt)))
    elif source == 'custom':
        path = Path(config['path'])
        receipt.update(input_sha256=sha256_path(path), targets='user-provided', selection='file order')
        inputs.append(('custom', 'custom', custom_records(path), config['max_examples'], dict(receipt)))
    elif source == 'manifest':
        records = [r for r in result['rows'] if r.get('kind') == 'utility' and r.get('split') == 'train']
        inputs.append(('manifest', 'custom', records, max(1, len(records)), {'source': 'manifest'}))
    else:
        raise ValueError('Unknown general data source: ' + source)
    for extra in additional:
        path = Path(extra['path'])
        inputs.append((extra['name'], 'custom', custom_records(path), extra['max_examples'],
                       {'source': 'custom', 'input_sha256': sha256_path(path),
                        'targets': 'user-provided', 'selection': 'file order'}))
    excluded = {digest(p.strip()) for p in excluded_prompts}
    seen, selected, skipped, sources = set(), [], Counter(), []
    for name, kind, records, limit, provenance in inputs:
        local_skipped, local_rows, inspected = Counter(), [], 0
        scan_limit = max(1000, limit * 100) if kind == 'dolci' else None
        if scan_limit is not None:
            records = islice(records, scan_limit)
            provenance['scan_limit'] = scan_limit
        for record in records:
            inspected += 1
            row = dolci_pair(record) if kind == 'dolci' else custom_pair(record)
            if row is None:
                local_skipped['unsupported_or_safety'] += 1
                continue
            if not all(isinstance(row.get(k), str) and row[k].strip() for k in ('prompt', 'target')):
                local_skipped['empty_text'] += 1
                continue
            key = digest(row['prompt'].strip())
            if key in excluded or key in seen:
                local_skipped['excluded_prompt' if key in excluded else 'duplicate_prompt'] += 1
                continue
            seen.add(key)
            item = {'id': 'general:' + key, 'source': 'general', 'split': 'train', 'kind': 'utility',
                    'prompt': row['prompt'], 'target': row['target'],
                    'original_id': row.get('original_id', row.get('id')), 'general_source': name}
            local_rows.append(item)
            if len(local_rows) >= limit:
                break
        selected.extend(local_rows)
        skipped.update(local_skipped)
        sources.append(dict(provenance, name=name, max_examples=limit, inspected=inspected,
                            count=len(local_rows), skipped=dict(local_skipped), rows_sha256=digest(local_rows)))
    if not selected:
        raise ValueError('No usable general-task examples after filtering')
    result['rows'] = [r for r in result['rows'] if not (r.get('kind') == 'utility' and
                       r.get('split') == 'train')] + selected
    receipt.update(count=len(selected), skipped=dict(skipped), rows_sha256=digest(selected), sources=sources)
    return result, receipt

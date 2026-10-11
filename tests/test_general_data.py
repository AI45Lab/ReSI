import json
from pathlib import Path
import tempfile
import unittest

from r2ai_autoresearch.research.config import load, validate
from r2ai_autoresearch.research.general_data import prepare_general, dolci_pair

ROOT = Path(__file__).resolve().parents[1]


class GeneralDataTests(unittest.TestCase):
    def setUp(self):
        self.config = load(ROOT / 'config.json')['data']['preparation']['general']
        self.training = {'rows': [
            {'id': 'a', 'source': 'attack', 'split': 'train', 'kind': 'harmful', 'prompt': 'attack'},
            {'id': 'old', 'source': 'general', 'split': 'train', 'kind': 'utility', 'prompt': 'old', 'target': 'old'}]}

    def test_dolci_filtering_pinning_and_target_preservation(self):
        def record(prompt, **kw):
            return dict(id=prompt, domain='Math', messages=[{'role': 'user', 'content': prompt},
                {'role': 'assistant', 'content': 'Reference answer'}], **kw)
        records = [record('heldout'), record('one'), record('one'), record('two')]
        safety = record('safety'); safety['domain'] = 'Safety'; records.insert(0, safety)
        captured = {}
        class Stream:
            def shuffle(self, **kwargs):
                captured.update(kwargs)
                return iter(records)
        def loader(dataset, **kwargs):
            captured.update(dataset=dataset, **kwargs)
            return Stream()
        self.config['max_examples'] = 2
        result, receipt = prepare_general(self.training, self.config, ['heldout'], loader=loader)
        self.assertEqual([r['prompt'] for r in result['rows']], ['attack', 'one', 'two'])
        self.assertEqual(result['rows'][-1]['target'], 'Reference answer')
        self.assertEqual(captured['revision'], self.config['revision'])
        self.assertTrue(captured['streaming'])
        self.assertEqual(receipt['skipped'], {'unsupported_or_safety': 1, 'excluded_prompt': 1, 'duplicate_prompt': 1})
        self.assertEqual(self.training['rows'][-1]['prompt'], 'old')

    def test_custom_json_and_jsonl_replace_dolci_without_loader(self):
        with tempfile.TemporaryDirectory() as d:
            for suffix in ('json', 'jsonl'):
                path = Path(d) / ('custom.' + suffix)
                row = {'prompt': 'My task', 'target': 'My answer'}
                path.write_text(json.dumps([row] if suffix == 'json' else row))
                cfg = dict(self.config, source='custom', path=str(path))
                def fail(*args, **kwargs):
                    raise AssertionError('Custom input must not access DOLCI')
                result, receipt = prepare_general(self.training, cfg, [], loader=fail)
                self.assertEqual(result['rows'][-1]['target'], 'My answer')
                self.assertEqual(receipt['count'], 1)
                self.assertIn('input_sha256', receipt)
                path.write_text(json.dumps([{'prompt': 'missing target'}]) if suffix == 'json' else '{"prompt":"missing target"}')
                with self.assertRaisesRegex(ValueError, 'prompt and target'):
                    prepare_general(self.training, cfg, [])

    def test_manifest_preserves_existing_utility(self):
        result, receipt = prepare_general(self.training, dict(self.config, source='manifest'), [])
        self.assertEqual(result, self.training)
        self.assertEqual(receipt['count'], 1)

    def test_no_silent_truncation_of_chat_or_tools(self):
        for messages in ([{'role': 'user'}, {'role': 'assistant'}, {'role': 'user'}],
                         [{'role': 'user', 'functions': 'tool'}, {'role': 'assistant'}]):
            self.assertIsNone(dolci_pair({'messages': messages}))

    def test_multiple_sources_deduplicate_exclude_and_preserve_provenance(self):
        with tempfile.TemporaryDirectory() as d:
            one, two = Path(d) / 'one.json', Path(d) / 'two.jsonl'
            one.write_text(json.dumps([{'prompt': 'unique', 'target': 'first'},
                                       {'prompt': 'heldout', 'target': 'do not train'}]))
            two.write_text('\n'.join(json.dumps(r) for r in [
                {'prompt': 'unique', 'target': 'conflicting second'},
                {'messages': [{'role': 'user', 'content': 'chat'}, {'role': 'assistant', 'content': 'answer'}]},
                {'prompt': 'beyond cap', 'target': 'not selected'}]))
            cfg = dict(self.config, source='custom', path=str(one), additional_sources=[
                {'name': 'user_math', 'path': str(two), 'max_examples': 1}])
            result, receipt = prepare_general(self.training, cfg, ['heldout'])
            self.assertEqual([r['prompt'] for r in result['rows']], ['attack', 'unique', 'chat'])
            self.assertEqual(result['rows'][1]['target'], 'first')
            self.assertEqual(result['rows'][-1]['general_source'], 'user_math')
            self.assertEqual([r['count'] for r in receipt['sources']], [1, 1])
            self.assertEqual(receipt['skipped'], {'excluded_prompt': 1, 'duplicate_prompt': 1})
            self.assertTrue(all(r['input_sha256'] for r in receipt['sources']))

    def test_dolci_and_custom_are_both_retained_with_independent_caps(self):
        class Stream:
            def shuffle(self, **kwargs):
                return iter([{'domain': 'Math', 'messages': [
                    {'role': 'user', 'content': 'dolci'}, {'role': 'assistant', 'content': 'answer'}]}])
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'extra.json'; path.write_text('[{"prompt":"custom","target":"answer"}]')
            cfg = dict(self.config, max_examples=1, additional_sources=[{'name': 'extra', 'path': str(path), 'max_examples': 1}])
            result, receipt = prepare_general(self.training, cfg, [], loader=lambda *a, **k: Stream())
            self.assertEqual([r['prompt'] for r in result['rows']], ['attack', 'dolci', 'custom'])
            self.assertEqual(receipt['count'], 2)
            self.assertEqual(receipt['sources'][0]['scan_limit'], 1000)

    def test_dolci_invalid_stream_is_bounded(self):
        class Stream:
            def shuffle(self, **kwargs):
                while True:
                    yield {'domain': 'Safety'}
        with self.assertRaisesRegex(ValueError, 'No usable'):
            prepare_general(self.training, dict(self.config, max_examples=1), [], loader=lambda *a, **k: Stream())

    def test_custom_chat_rejects_context_loss(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'chat.json'
            cfg = dict(self.config, source='custom', path=str(path))
            for messages in ([{'role': 'system', 'content': 'important'}, {'role': 'user', 'content': 'q'}, {'role': 'assistant', 'content': 'a'}],
                             [{'role': 'user', 'content': 'q'}, {'role': 'assistant', 'content': 'a', 'tool_calls': [{}]}]):
                path.write_text(json.dumps([{'messages': messages}]))
                with self.assertRaisesRegex(ValueError, 'context must not be dropped'):
                    prepare_general(self.training, cfg, [])

    def test_manifest_with_extra_local_dataset_no_download(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'extra.json'; path.write_text('[{"prompt":"new","target":"answer"}]')
            cfg = dict(self.config, source='manifest', additional_sources=[{'name': 'extra', 'path': str(path), 'max_examples': 2}])
            result, receipt = prepare_general(self.training, cfg, [])
            self.assertEqual([r['prompt'] for r in result['rows']], ['attack', 'old', 'new'])
            self.assertEqual(receipt['count'], 2)
            path.write_text('[{"prompt":"heldout","target":"answer"}]')
            with self.assertRaisesRegex(ValueError, 'No usable'):
                prepare_general({'rows': []}, cfg, ['heldout'])

    def test_additional_source_validation_and_path_resolution(self):
        c = json.loads((ROOT / 'config.json').read_text())
        c['data']['preparation']['general']['additional_sources'] = [
            {'name': 'extra', 'path': 'extra.jsonl', 'max_examples': 30}]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'config.json'; path.write_text(json.dumps(c))
            self.assertEqual(load(path)['data']['preparation']['general']['additional_sources'][0]['path'], str((Path(d) / 'extra.jsonl').resolve()))
        c['data']['preparation']['general']['additional_sources'] *= 2
        with self.assertRaisesRegex(ValueError, 'unique'): validate(c)

    def test_general_config_validation_and_relative_paths(self):
        c = json.loads((ROOT / 'config.json').read_text())
        g = c['data']['preparation']['general']
        g['revision'] = 'main'
        with self.assertRaisesRegex(ValueError, 'Pin DOLCI'): validate(c)
        g['source'] = 'custom'
        with self.assertRaisesRegex(ValueError, 'general.path'): validate(c)
        g['path'] = 'my_data.jsonl'
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'config.json'; path.write_text(json.dumps(c))
            self.assertEqual(load(path)['data']['preparation']['general']['path'], str((Path(d) / 'my_data.jsonl').resolve()))


if __name__ == '__main__':
    unittest.main()

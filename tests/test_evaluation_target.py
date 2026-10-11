"""Checkpoint binding and real local HTTP lifecycle, without model-performance claims."""
from pathlib import Path
import copy,json,socket,sys,tempfile,unittest
from unittest.mock import patch
from r2ai_autoresearch.executor import sha256_path
from r2ai_autoresearch.research.evaluation_target import selected_checkpoint,local_endpoint,model_service

ROOT=Path(__file__).resolve().parents[1]

def fixture(root):
    c=json.loads((ROOT/'config.smoke.json').read_text())
    model=root/'model';model.mkdir();(model/'config.json').write_text('{"model_type":"qwen3"}');(model/'model.safetensors').write_bytes(b'fixture-not-model-weights')
    run=root/'search';run.mkdir();(run/'summary.json').write_text(json.dumps({'status':'completed','retained_model':str(model),'retained_model_sha256':sha256_path(model),'history':[{'selected':'fixture'}]}))
    return c,model,run

class EvaluationTargetTests(unittest.TestCase):
    def test_cached_export_rejects_mutated_model_content(self):
        from r2ai_autoresearch.research.export import export_checkpoint,exported_files
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);c,model,run=fixture(root);output=root/'exported';output.mkdir()
            (output/'config.json').write_text('{"model_type":"qwen3"}');(output/'model.safetensors').write_bytes(b'export-fixture')
            receipt={'identity':{'source_sha256':sha256_path(model),'settings':c['model']},'exported_files':exported_files(output)}
            (output/'r2ai_export.json').write_text(json.dumps(receipt))
            self.assertEqual(export_checkpoint(model,output,c['model']),str(output.resolve()))
            (output/'model.safetensors').write_bytes(b'mutated export')
            with self.assertRaisesRegex(ValueError,'content changed'):export_checkpoint(model,output,c['model'])

    def test_completed_search_binds_exact_checkpoint_and_detects_changes(self):
        with tempfile.TemporaryDirectory() as t:
            c,model,run=fixture(Path(t));resolved,record=selected_checkpoint(c,run)
            self.assertEqual(resolved['model']['path'],str(model.resolve()));self.assertEqual(record['completed_rounds'],1)
            self.assertIsNone(c['model']['path'])
            (model/'model.safetensors').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'changed'):selected_checkpoint(c,run)

    def test_incomplete_search_cannot_enter_final_evaluation(self):
        with tempfile.TemporaryDirectory() as t:
            c,model,run=fixture(Path(t));(run/'status.json').write_text('{"status":"failed"}')
            with self.assertRaisesRegex(ValueError,'not completed'):selected_checkpoint(c,run)
            (run/'status.json').unlink();(run/'summary.json').write_text('{"status":"running"}')
            with self.assertRaisesRegex(ValueError,'must complete'):selected_checkpoint(c,run)

    def test_local_mode_rejects_external_endpoints_and_occupied_ports(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);c,model,run=fixture(root);c['model']['path']=str(model)
            c['final_evaluation']['target']['base_url']='https://example.test/v1'
            with self.assertRaisesRegex(ValueError,'localhost'):local_endpoint(c)
            with socket.socket() as occupied:
                occupied.bind(('127.0.0.1',0));occupied.listen();port=occupied.getsockname()[1]
                c['final_evaluation']['target']['base_url']=f'http://127.0.0.1:{port}/v1'
                with self.assertRaisesRegex(ValueError,'occupied'):
                    with model_service(c,{'checkpoint':str(model)},root):pass

    def test_external_mode_records_binding_without_loading_or_launching(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);c,model,run=fixture(root)
            with patch('r2ai_autoresearch.research.serve.serving_argv',side_effect=AssertionError('no local launch')):
                with model_service(c,{'checkpoint':str(model)},root,external=True) as supplied:self.assertIs(supplied,c)
            record=json.loads((root/'target_checkpoint.json').read_text())
            self.assertFalse(record['endpoint_checkpoint_verified']);self.assertEqual(record['serving'],'external')

    def test_check_uses_configured_search_output_without_starting_service(self):
        from r2ai_autoresearch.research.full_evaluate import main
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);c,model,run=fixture(root)
            c['runtime']['output']=str(run);c['final_evaluation']['benchmarks']=['gpqa_diamond']
            config=root/'config.json';config.write_text(json.dumps(c))
            row={'id':'fixture','benchmark':'gpqa_diamond','prompt':'fixture','reference':'A','metadata':{'domain':'fixture'}}
            with patch('sys.argv',['evaluate','--config',str(config),'--run','--check']), \
                 patch('r2ai_autoresearch.research.full_evaluate.Client.endpoint',return_value=('http://example.test/v1','judge','')), \
                 patch('r2ai_autoresearch.formal_eval.data.load_benchmark',return_value=[row]), \
                 patch.dict('r2ai_autoresearch.formal_eval.data.EXPECTED_COUNTS',{'gpqa_diamond':1}), \
                 patch('r2ai_autoresearch.research.evaluation_target.model_service',side_effect=AssertionError('preflight must not launch')):
                main()

    def test_full_evaluation_cli_reads_retained_model_and_writes_scores(self):
        from r2ai_autoresearch.research.full_evaluate import main
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);c,model,run=fixture(root)
            with socket.socket() as socket_:
                socket_.bind(('127.0.0.1',0));port=socket_.getsockname()[1]
            c['final_evaluation']['benchmarks']=['gpqa_diamond']
            c['final_evaluation']['target']['base_url']=f'http://127.0.0.1:{port}/v1'
            c['final_evaluation']['output']=str(root/'evaluation')
            config=root/'config.json';config.write_text(json.dumps(c))
            server=root/'server.py';requests=root/'requests.jsonl'
            server.write_text(r'''from http.server import HTTPServer,BaseHTTPRequestHandler
import sys,json
class H(BaseHTTPRequestHandler):
 def do_GET(self):self.send_response(200);self.end_headers();self.wfile.write(b"ok")
 def do_POST(self):
  request=json.loads(self.rfile.read(int(self.headers["Content-Length"])))
  with open(sys.argv[2],"a") as f:f.write(json.dumps(request)+"\n")
  response={"choices":[{"message":{"content":"\\boxed{A}"},"finish_reason":"stop"}]}
  self.send_response(200);self.send_header("Content-Type","application/json");self.end_headers();self.wfile.write(json.dumps(response).encode())
HTTPServer(("127.0.0.1",int(sys.argv[1])),H).serve_forever()
''')
            row={'id':'fixture','benchmark':'gpqa_diamond','prompt':'Engineering fixture question','reference':'A','metadata':{'domain':'fixture','category':'fixture'}}
            with patch.dict('os.environ',{'R2AI_SERVING_PYTHON':sys.executable}), \
                 patch('sys.argv',['evaluate','--config',str(config),'--run',str(run)]), \
                 patch('r2ai_autoresearch.research.full_evaluate.Client.endpoint',return_value=('http://example.test/v1','judge','')), \
                 patch('r2ai_autoresearch.formal_eval.data.load_benchmark',return_value=[row]), \
                 patch.dict('r2ai_autoresearch.formal_eval.data.EXPECTED_COUNTS',{'gpqa_diamond':1}), \
                 patch('r2ai_autoresearch.research.serve.serving_argv',return_value=[sys.executable,str(server),str(port),str(requests)]):
                main()
            output=root/'evaluation';metrics=json.loads((output/'summary.json').read_text())
            self.assertEqual(metrics['gpqa_diamond']['status'],'complete')
            self.assertEqual(metrics['gpqa_diamond']['metrics']['accuracy'],1.)
            record=json.loads((output/'target_checkpoint.json').read_text())
            self.assertEqual(record['checkpoint_sha256'],sha256_path(model))
            self.assertEqual(json.loads(requests.read_text())['model'],'resi-retained')
            self.assertTrue((output/'metrics.csv').is_file())
            self.assertEqual(json.loads((output/'identity.json').read_text())['checkpoint']['checkpoint'],str(model.resolve()))

    def test_local_http_service_is_stopped_after_evaluation_failure(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);c,model,run=fixture(root);c['model']['path']=str(model)
            with socket.socket() as socket_:
                socket_.bind(('127.0.0.1',0));port=socket_.getsockname()[1]
            c['final_evaluation']['target']['base_url']=f'http://127.0.0.1:{port}/v1'
            c['final_evaluation']['target']['model']='retained-fixture'
            server=root/'server.py';server.write_text('''from http.server import HTTPServer,BaseHTTPRequestHandler
import sys
class H(BaseHTTPRequestHandler):
 def do_GET(self):self.send_response(200);self.end_headers();self.wfile.write(b"ok")
HTTPServer(("127.0.0.1",int(sys.argv[1])),H).serve_forever()
''')
            with patch.dict('os.environ',{'R2AI_SERVING_PYTHON':sys.executable}), \
                 patch('r2ai_autoresearch.research.serve.serving_argv',return_value=[sys.executable,str(server),str(port)]):
                with self.assertRaisesRegex(RuntimeError,'fixture failure'):
                    with model_service(c,{'checkpoint':str(model)},root,startup_timeout=10):raise RuntimeError('fixture failure')
            with socket.socket() as probe:self.assertNotEqual(probe.connect_ex(('127.0.0.1',port)),0)
            record=json.loads((root/'target_checkpoint.json').read_text());self.assertEqual(record['served_checkpoint'],str(model))

if __name__=='__main__':unittest.main()

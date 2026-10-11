"""Serve an exported checkpoint for the formal evaluator."""
import argparse
import json
import os
from pathlib import Path
from urllib.parse import urlparse
from .config import load
from .model_adapter import validate_checkpoint


def serving_argv(c,model):
    model=validate_checkpoint(model)
    if (model/'r2ai_checkpoint.json').exists():raise ValueError('Export the selected checkpoint with scripts/export_model.sh first')
    target=c['final_evaluation']['target'];address=urlparse(target['base_url'] or '')
    if address.scheme!='http' or address.hostname not in ('localhost','127.0.0.1'):
        raise ValueError('For local serving set final_evaluation.target.base_url to http://127.0.0.1:PORT/v1')
    if not target['model']:raise ValueError('Set final_evaluation.target.model to the served model name')
    python=os.getenv('R2AI_SERVING_PYTHON',str(Path(c['training']['grpo']['runtime_path'])/'.venv/bin/python'))
    argv=[python,'-m','vllm.entrypoints.openai.api_server','--model',str(model),'--served-model-name',target['model'],
          '--host','127.0.0.1','--port',str(address.port or 80),'--tensor-parallel-size',str(c['training']['grpo']['tensor_parallel_size']),
          '--dtype',c['model']['dtype']]
    if c['model']['trust_remote_code']:argv.append('--trust-remote-code')
    family=json.loads((model/'config.json').read_text()).get('model_type')
    if family=='mistral3':argv.extend(['--tokenizer-mode','mistral','--config-format','mistral','--reasoning-parser','mistral'])
    return argv


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--model',required=True)
    p.add_argument('--check',action='store_true');a=p.parse_args();c=load(a.config)
    argv=serving_argv(c,a.model);python=argv[0]
    if a.check:print(json.dumps({'argv':argv,'gpu_ids':c['runtime']['gpu_ids']}));return
    if not Path(python).is_file():raise ValueError('Run scripts/setup_verl.sh or set R2AI_SERVING_PYTHON to a vLLM environment')
    os.environ['CUDA_VISIBLE_DEVICES']=c['runtime']['gpu_ids']
    os.execv(python,argv)

if __name__=='__main__':main()

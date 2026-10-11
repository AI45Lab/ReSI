"""Export a selected adapter lineage as a standalone Hugging Face checkpoint."""
import argparse
import gc
import json
from pathlib import Path
from ..executor import sha256_path
from ..portable.contracts import save


def exported_files(directory):
    directory=Path(directory)
    return {str(p.relative_to(directory)):sha256_path(p) for p in directory.rglob('*') if p.is_file() and p.name!='r2ai_export.json'}


def export_checkpoint(model_path, output, settings):
    from .model_adapter import load_model, validate_checkpoint, copy_model_assets
    source=validate_checkpoint(model_path);dest=Path(output).resolve()
    identity={'source_sha256':sha256_path(source),'settings':settings}
    receipt=dest/'r2ai_export.json'
    if receipt.is_file():
        record=json.loads(receipt.read_text())
        if record['identity']!=identity:raise ValueError('Export destination belongs to another checkpoint')
        validate_checkpoint(dest)
        if not record.get('exported_files'):raise ValueError('Legacy export has no integrity manifest; export into a new directory')
        if exported_files(dest)!=record['exported_files']:raise ValueError('Exported model content changed')
        return str(dest)
    if dest.exists() and any(dest.iterdir()):raise ValueError('Export destination must be empty or a completed matching export')
    from transformers import AutoTokenizer
    from peft import PeftModel
    import torch
    dest.mkdir(parents=True,exist_ok=True)
    model,_=load_model(source,settings)
    if isinstance(model,PeftModel):model=model.merge_and_unload(safe_merge=True)
    model.save_pretrained(dest,safe_serialization=True,max_shard_size='5GB')
    AutoTokenizer.from_pretrained(source,local_files_only=True).save_pretrained(dest)
    model.generation_config.save_pretrained(dest)
    copy_model_assets(source,dest)
    del model;gc.collect();torch.cuda.empty_cache()
    validate_checkpoint(dest)
    save(receipt,{'identity':identity,'exported_files':exported_files(dest),'format':'standalone_huggingface','parent_required':False,
                  'validation':'weight files complete; run evaluation for inference acceptance'})
    return str(dest)


def main():
    from .config import load
    import os
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    g=p.add_mutually_exclusive_group(required=True);g.add_argument('--model');g.add_argument('--run')
    p.add_argument('--output',required=True);a=p.parse_args();c=load(a.config)
    os.environ['CUDA_VISIBLE_DEVICES']=c['runtime']['gpu_ids']
    model=a.model or json.loads((Path(a.run)/'summary.json').read_text())['retained_model']
    print(json.dumps({'model':export_checkpoint(model,a.output,c['model'])}))

if __name__=='__main__':main()

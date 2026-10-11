"""Bind formal evaluation to a retained checkpoint and manage its local service."""
from pathlib import Path
from urllib.parse import urlparse
from contextlib import contextmanager
import copy,json,os,socket,subprocess,signal,time,urllib.request
from ..executor import sha256_path
from ..portable.contracts import save
from .model_adapter import validate_checkpoint


def selected_checkpoint(config,run=None,checkpoint=None):
    if run:
        directory=Path(run).resolve();path=directory/'summary.json'
        if not path.is_file():raise ValueError('Search has no completed summary: '+str(directory))
        result=json.loads(path.read_text())
        if result.get('status')!='completed':raise ValueError('Search must complete before final evaluation')
        status=directory/'status.json'
        if status.is_file() and json.loads(status.read_text()).get('status')!='completed':
            raise ValueError('Search status is not completed')
        model=Path(result['retained_model']).resolve()
        expected=result.get('retained_model_sha256')
        if not expected:raise ValueError('Search summary lacks checkpoint integrity information')
        actual=sha256_path(model)
        if actual!=expected:raise ValueError('Retained checkpoint changed after search')
        provenance={'search_run':str(directory),'checkpoint':str(model),'checkpoint_sha256':actual,
                    'completed_rounds':len(result.get('history',[]))}
    else:
        model=Path(checkpoint).resolve();provenance={'checkpoint':str(model),'checkpoint_sha256':sha256_path(model)}
    validate_checkpoint(model)
    c=copy.deepcopy(config);c['model']['path']=str(model)
    return c,provenance


def local_endpoint(config):
    c=copy.deepcopy(config);target=c['final_evaluation']['target']
    if not target['base_url']:target['base_url']='http://127.0.0.1:8000/v1'
    if not target['model']:target['model']='resi-retained'
    address=urlparse(target['base_url'])
    if address.scheme!='http' or address.hostname not in ('localhost','127.0.0.1') or address.path.rstrip('/')!='/v1':
        raise ValueError('Automatic evaluation requires a localhost /v1 endpoint; use --external-target for a managed service')
    if not address.port or not 1024<=address.port<=65535:raise ValueError('Choose an explicit local serving port between 1024 and 65535')
    return c


@contextmanager
def model_service(config,provenance,dest,*,external=False,export_dir=None,startup_timeout=600):
    if external:
        save(dest/'target_checkpoint.json',{**provenance,'serving':'external','endpoint_checkpoint_verified':False})
        yield config;return
    from .serve import serving_argv
    c=local_endpoint(config);source=Path(c['model']['path']);address=urlparse(c['final_evaluation']['target']['base_url'])
    with socket.socket() as probe:
        probe.settimeout(1)
        if probe.connect_ex((address.hostname,address.port))==0:
            raise ValueError('Local evaluation port is occupied; refusing to reuse an unknown model service')
    interpreter=os.getenv('R2AI_SERVING_PYTHON',str(Path(c['training']['grpo']['runtime_path'])/'.venv/bin/python'))
    if not Path(interpreter).is_file():
        raise ValueError('Install scripts/setup_verl.sh or set R2AI_SERVING_PYTHON to a vLLM interpreter for local evaluation')
    os.environ['CUDA_VISIBLE_DEVICES']=c['runtime']['gpu_ids']
    if (source/'r2ai_checkpoint.json').is_file():
        from .export import export_checkpoint
        source=Path(export_checkpoint(source,export_dir or dest/'exported-model',c['model']))
    c['model']['path']=str(source);argv=serving_argv(c,source)
    save(dest/'target_checkpoint.json',{**provenance,'serving':'local','served_checkpoint':str(source),
                                      'served_model_name':c['final_evaluation']['target']['model']})
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    log=(dest/'model-service.log').open('a')
    process=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,
                             env={**os.environ,'CUDA_VISIBLE_DEVICES':c['runtime']['gpu_ids']},start_new_session=True)
    try:
        deadline=time.monotonic()+startup_timeout;healthy=False
        url=c['final_evaluation']['target']['base_url'].rsplit('/v1',1)[0]+'/health'
        while time.monotonic()<deadline:
            if process.poll() is not None:raise RuntimeError('Model service exited; inspect '+str(dest/'model-service.log'))
            try:
                with opener.open(url,timeout=2) as response:healthy=response.status==200
            except (OSError,urllib.error.URLError):pass
            if healthy:break
            time.sleep(.5)
        if not healthy:raise RuntimeError('Model service startup timed out; inspect '+str(dest/'model-service.log'))
        yield c
    finally:
        if process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=10)
        log.close()

"""Separate crossed-controls pilot; sidecar capture without changing runner interfaces."""
import os
os.environ['HF_HUB_ENABLE_HF_TRANSFER']='0'
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]


def main():
    import torch
    import yaml
    from deploy.runpod.smoke import require_remote_gpu, MODEL_ID, REVISION
    from deploy.runpod.position_capture import resolve_positions, PositionCapture, compare_capture
    from robot_refusal.model_setup import setup_model
    from robot_refusal.models.qwen_vl import QwenVLAdapter
    from robot_refusal.records import ModelConfig
    from src.eval.integrated import run_integrated
    from transformers import StoppingCriteria, StoppingCriteriaList

    require_remote_gpu(torch)
    freeze=json.loads((ROOT/'configs/crossed_controls_v1.freeze.json').read_text())
    for name,digest in freeze['sha256'].items():
        if hashlib.sha256((ROOT/name).read_text(encoding='utf-8').encode()).hexdigest()!=digest:
            raise ValueError('Frozen input differs: '+name)
    output=Path('/workspace/results');output.mkdir(exist_ok=False)
    (output/'freeze-verification.json').write_text(json.dumps(freeze,indent=2))
    sidecars=output/'positions';sidecars.mkdir()
    snapshot=setup_model(MODEL_ID,'/workspace/model-cache',revision=REVISION)
    cfg=yaml.safe_load((ROOT/'configs/crossed_controls_v1.yaml').read_text())
    model=dict(cfg['model_config'],model_dir=snapshot['snapshot_path'])
    cfg.update(model_config=model,run_dir=str(output/'pilot'))
    owner=QwenVLAdapter()
    original=None
    deadline=time.monotonic()+1200
    checked=set()
    class Deadline(StoppingCriteria):
        def __call__(self,*args,**kwargs):
            if time.monotonic()>deadline:raise TimeoutError('Twenty-minute inference limit')
            return False
    class Session:
        backend='qwen'
        def load(self,c):
            if c!=ModelConfig.model_validate(model):raise ValueError('Model config changed')
        def prepare(self,messages,generation_config):
            text=messages[-1]['content'][-1]['text']
            self.instruction=text.split('Please give me step-by-step instructions for this request: ')[-1]
            self.condition='text_chat' if len(messages)==1 else 'text_tools'
            return owner.prepare(messages,generation_config)
        def generate(self,prepared,capture_config,intervention=None):
            if intervention is not None:raise ValueError('Interventions off')
            positions=resolve_positions(prepared,self.instruction)
            key=hashlib.sha256(json.dumps([prepared.metadata['prompt_token_ids'],prepared.generation_config],sort_keys=True).encode()).hexdigest()
            folder=sidecars/key;folder.mkdir(exist_ok=False)
            result=[]
            def generate():
                value=owner.generate(prepared,capture_config,None)
                result.append(value)
                return value
            if self.condition not in checked:
                observed,check=compare_capture(owner,prepared,positions,generate)
                (folder/'equality.json').write_text(json.dumps(check,indent=2))
                # Fail closed before accepting an empirically mismatched capture.
                if not check['allclose']:
                    raise ValueError('Independent forward/capture equality failed')
                checked.add(self.condition)
            else:
                observed=PositionCapture(owner.modules,owner.hidden_size,prepared,positions)
                with observed:generate()
            entries={}
            for name in positions:
                tensor=observed.stack(name)
                if list(tensor.shape)!=[len(owner.modules),owner.hidden_size] or not torch.isfinite(tensor).all():
                    raise ValueError('Invalid named capture')
                path=folder/(name+'.pt');torch.save(tensor,path)
                entries[name]=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),shape=list(tensor.shape))
            value=result[0]
            # Compare final-prompt sidecar against the unchanged original runner hooks.
            legacy=torch.stack([value.activations[f'layer_{i}.prompt_{positions["prompt_end"]}.residual'] for i in range(len(owner.modules))])
            if not torch.equal(legacy,observed.stack('prompt_end')):raise ValueError('Legacy boundary capture differs')
            meta=dict(positions=positions,entries=entries,condition=self.condition,
                instruction=self.instruction,generation=prepared.generation_config,
                prompt_token_ids_sha256=hashlib.sha256(json.dumps(prepared.metadata['prompt_token_ids']).encode()).hexdigest(),
                capture=observed.metadata(),legacy_prompt_equal=True)
            (folder/'metadata.json').write_text(json.dumps(meta,indent=2))
            value.resolved_configuration['position_sidecar']=str(folder/'metadata.json')
            return value
        def close(self):pass
    try:
        owner.load(ModelConfig.model_validate(model))
        original=owner.model.generate
        def bounded(*args,**kwargs):
            kwargs['stopping_criteria']=StoppingCriteriaList([*kwargs.get('stopping_criteria',[]),Deadline()])
            return original(*args,**kwargs)
        owner.model.generate=bounded
        run_integrated(cfg,adapter_factory=Session,adapter_factory_is_test=False)
        rows=[json.loads(s) for s in (output/'pilot/results.jsonl').read_text().splitlines()]
        links=[]
        for row in rows:
            if row.get('error') or row['truncated'] or not row['text'].strip():raise ValueError('Incomplete trial '+row['task_id'])
            raw=json.loads(Path(row['attempts'][-1]['trial_path']).read_text())
            meta=raw['turns'][0]['resolved_model_config']['position_sidecar']
            links.append(dict(run_key=row['run_key'],task_id=row['task_id'],condition=row['condition'],position_sidecar=meta))
        report=dict(trials=len(rows),valid_complete=len(links),shape=[len(owner.modules),owner.hidden_size],checkpoint=REVISION,
                    equality_conditions=sorted(checked),legacy_capture_unchanged=True)
        (output/'position_links.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in links))
        (output/'validation.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report),flush=True)
        if len(rows)!=16 or len(checked)!=2:raise ValueError('Incomplete crossed design')
    finally:
        if original is not None:owner.model.generate=original
        owner.close()


if __name__=='__main__':main()

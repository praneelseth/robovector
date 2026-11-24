"""Bounded exploratory five-pair run; lifecycle shutdown is managed externally."""
import os
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]


def main():
    import torch
    import yaml
    from deploy.runpod.smoke import require_remote_gpu, MODEL_ID, REVISION
    from robot_refusal.model_setup import setup_model
    from robot_refusal.models.qwen_vl import QwenVLAdapter
    from robot_refusal.records import ModelConfig
    from src.eval.integrated import run_integrated
    from transformers import StoppingCriteria, StoppingCriteriaList

    require_remote_gpu(torch)
    freeze=json.loads((ROOT/'configs/pilot_five_pairs_v1.freeze.json').read_text())
    for name,digest in freeze['sha256'].items():
        if hashlib.sha256((ROOT/name).read_text(encoding='utf-8').encode()).hexdigest()!=digest:
            raise ValueError('Frozen input differs: '+name)
    output=Path('/workspace/results');output.mkdir(exist_ok=False)
    (output/'freeze-verification.json').write_text(json.dumps(freeze,indent=2))
    snapshot=setup_model(MODEL_ID,'/workspace/model-cache',revision=REVISION)
    cfg=yaml.safe_load((ROOT/'configs/pilot_five_pairs_v1.yaml').read_text())
    model=dict(cfg['model_config'],model_dir=snapshot['snapshot_path'])
    cfg.update(model_config=model,run_dir=str(output/'pilot'))
    owner=QwenVLAdapter()
    original=None
    deadline=time.monotonic()+1200
    class Deadline(StoppingCriteria):
        def __call__(self,*args,**kwargs):
            if time.monotonic()>deadline:raise TimeoutError('Twenty-minute inference limit')
            return False
    class Session:
        backend='qwen'
        def load(self,c):
            if c!=ModelConfig.model_validate(model):raise ValueError('Model config changed')
        def prepare(self,*args,**kwargs):return owner.prepare(*args,**kwargs)
        def generate(self,*args,**kwargs):return owner.generate(*args,**kwargs)
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
        good=0
        for row in rows:
            if row.get('error') or row['truncated'] or not row['text'].strip():continue
            t=torch.load(row['activation_path'],weights_only=True,map_location='cpu')
            if list(t.shape)==[len(owner.modules),owner.hidden_size] and torch.isfinite(t).all():good+=1
        report=dict(trials=len(rows),valid_complete=good,shape=[len(owner.modules),owner.hidden_size],checkpoint=REVISION)
        (output/'validation.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report),flush=True)
        if len(rows)!=60 or good!=60:raise RuntimeError('Incomplete trials; preserve artifacts for review')
    finally:
        if original is not None:owner.model.generate=original
        owner.close()


if __name__=='__main__':main()

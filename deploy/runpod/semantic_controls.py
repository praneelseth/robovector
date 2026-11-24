"""Assessment-only crossed-scene diagnostic; no interventions or robot execution."""
import os
os.environ['HF_HUB_ENABLE_HF_TRANSFER']='0'
import hashlib
import json
from pathlib import Path
import sys
import time
from functools import partial

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]


def main():
    import torch
    from deploy.runpod.smoke import require_remote_gpu, MODEL_ID, REVISION
    from robot_refusal.model_setup import setup_model
    from robot_refusal.models.qwen_vl import QwenVLAdapter
    from robot_refusal.records import ModelConfig
    from robot_refusal.agent import run_trial
    from robot_refusal.artifacts import write_result
    from src.eval.integration import make_trial, stack_prompt_activations
    from src.eval.semantic_controls import build_semantic_messages, score_semantic_output
    from transformers import StoppingCriteria, StoppingCriteriaList

    require_remote_gpu(torch)
    freeze=json.loads((ROOT/'configs/semantic_controls_v1.freeze.json').read_text())
    for name,digest in freeze['sha256'].items():
        if hashlib.sha256((ROOT/name).read_text(encoding='utf-8').encode()).hexdigest()!=digest:
            raise ValueError('Frozen input differs: '+name)
    output=Path('/workspace/results');output.mkdir(exist_ok=False)
    (output/'freeze-verification.json').write_text(json.dumps(freeze,indent=2))
    snapshot=setup_model(MODEL_ID,'/workspace/model-cache',revision=REVISION)
    model=dict(model_id=MODEL_ID,model_dir=snapshot['snapshot_path'],device='cuda:0',dtype='bfloat16',attn_implementation='sdpa')
    owner=QwenVLAdapter();original=None
    deadline=time.monotonic()+900
    class Deadline(StoppingCriteria):
        def __call__(self,*args,**kwargs):
            if time.monotonic()>deadline:raise TimeoutError('Fifteen-minute inference cap')
            return False
    class Session:
        backend='qwen'
        def load(self,c):
            if c!=ModelConfig.model_validate(model):raise ValueError('Model config changed')
        def prepare(self,*args,**kwargs):return owner.prepare(*args,**kwargs)
        def generate(self,*args,**kwargs):return owner.generate(*args,**kwargs)
        def close(self):pass
    rows=[]
    try:
        owner.load(ModelConfig.model_validate(model))
        original=owner.model.generate
        def bounded(*args,**kwargs):
            kwargs['stopping_criteria']=StoppingCriteriaList([*kwargs.get('stopping_criteria',[]),Deadline()])
            return original(*args,**kwargs)
        owner.model.generate=bounded
        tasks=[json.loads(s) for s in (ROOT/'data/tasks_crossed_controls_v1.jsonl').read_text().splitlines()]
        for task in tasks:
            for condition in ['text_chat','text_tools']:
                for kind in ['target_state','action_danger']:
                    key=hashlib.sha256(json.dumps([task['task_id'],condition,kind,'semantic-v1']).encode()).hexdigest()
                    seed=(42+int(key[:8],16))%2**32
                    attempts=[]
                    for budget in [128,256,512]:
                        trial=make_trial(task,condition,trial_id=key,seed=seed,max_new_tokens=budget,temperature=.7,
                            model_config=model,image_paths=[])
                        result=run_trial(trial,Session(),prompt_builder=partial(build_semantic_messages,diagnostic_kind=kind))
                        folder=output/'trials'/key/str(budget)
                        write_result(result,folder)
                        attempts.append(dict(budget=budget,truncated=result.token_limit_reached,trial_path=str(folder/'result.json')))
                        if result.exceptions:raise ValueError('Runner error: '+str(result.exceptions))
                        hidden=stack_prompt_activations(result)
                        if list(hidden.shape)!=[36,4096] or not torch.isfinite(hidden).all():raise ValueError('Capture shape')
                        torch.save(hidden,folder/'prompt_layers.pt')
                        if result.token_limit_reached:continue
                        break
                    turn=result.turns[0]
                    score=score_semantic_output(turn.raw_output,condition,kind,task) if not result.token_limit_reached else dict(status='truncated',correct=None)
                    row=dict(run_key=key,task_id=task['task_id'],family=task['family_id'],arrangement=task['arrangement'],
                        target=task['target'],harm_label=task['harm_label'],condition=condition,diagnostic_kind=kind,seed=seed,
                        raw_output=turn.raw_output,attempts=attempts,truncated=result.token_limit_reached,
                        score=score,activation_shape=list(hidden.shape),activation_path=str(folder/'prompt_layers.pt'),
                        activation_sha256=hashlib.sha256((folder/'prompt_layers.pt').read_bytes()).hexdigest(),
                        proposed_call_records=[c.model_dump() for c in turn.calls],
                        trial_path=str(folder/'result.json'))
                    rows.append(row)
                    (output/'results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
                    print(json.dumps({k:row[k] for k in ['task_id','condition','diagnostic_kind','score']}),flush=True)
        from collections import Counter
        report=dict(trials=len(rows),truncated=sum(r['truncated'] for r in rows),
            scoring_status=dict(Counter(r['score']['status'] for r in rows)),checkpoint=REVISION,
            interpretation='Assessment-only prompts change the task; no causal inference about action-trial activations')
        (output/'validation.json').write_text(json.dumps(report,indent=2))
        if len(rows)!=32 or report['truncated']:raise ValueError('Incomplete assessment diagnostic')
    finally:
        if original is not None:owner.model.generate=original
        owner.close()


if __name__=='__main__':main()

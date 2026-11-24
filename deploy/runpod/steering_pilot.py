"""Frozen-candidate confirmation and small selective steering development pilot."""
import os
os.environ['HF_HUB_ENABLE_HF_TRANSFER']='0'
import hashlib,json,sys,time
from pathlib import Path
from contextlib import nullcontext
from functools import partial
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
FREEZE_PATTERN='configs/steering_{stage}_v1.freeze.json'
CONFIRMATION_PREFIX=None
TOKEN_BUDGETS=(1024,2048)
CONFIRMATION_SAFE_FIRST=False
ROBOT_JOB_MODE='full'


def restore_confirmation(path,held,stage):
    allowed={r['id']:r for r in held}
    rows=[json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines()]
    seen=set()
    for row in rows:
        record=row['record'];identifier=record['id']
        expected='baseline' if stage=='confirmation_baseline' else 'ablation' if record['harm_label']=='harmful' else 'add_0.25'
        if identifier in seen or allowed.get(identifier)!=record or row['arm']!=expected:
            raise ValueError('Confirmation checkpoint does not match frozen records/arm')
        if not isinstance(row.get('text'),str) or not isinstance(row.get('truncated'),bool):
            raise ValueError('Incomplete checkpoint row')
        seen.add(identifier)
    return rows


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['confirmation_baseline','confirmation_interventions','robot_pilot'],required=True)
    stage=parser.parse_args().stage
    import torch
    from transformers import StoppingCriteria,StoppingCriteriaList
    import transformers
    from deploy.runpod.smoke import require_remote_gpu,MODEL_ID,REVISION
    from robot_refusal.models.qwen_vl import QwenVLAdapter
    from robot_refusal.model_setup import setup_model
    from robot_refusal.records import ModelConfig,CaptureConfig
    from robot_refusal.agent import run_trial
    from robot_refusal.artifacts import write_result
    from src.analysis.interventions import AllPositionIntervention,seeded_random_direction
    from src.analysis.generation_quality import LoopStop
    from src.eval.integration import make_trial,build_messages,label_inputs,ACTION_NAMES
    from src.eval.labeling import classify_outcome
    require_remote_gpu(torch)
    freeze=json.loads((ROOT/FREEZE_PATTERN.format(stage=stage)).read_text())
    for p,digest in freeze['sha256'].items():
        raw=(ROOT/p).read_bytes() if p.endswith('.pt') else (ROOT/p).read_text(encoding='utf-8').encode()
        if hashlib.sha256(raw).hexdigest()!=digest:raise ValueError('Freeze mismatch '+p)
    chosen=json.loads((ROOT/'outputs/planning/steering-pilot-v1/selected.json').read_text())
    direction=torch.load(ROOT/chosen['direction_path'],weights_only=True,map_location='cpu')
    random=seeded_random_direction(direction,20261124)
    out=Path('/workspace/results');out.mkdir(exist_ok=False)
    (out/'stage.json').write_text(json.dumps(dict(stage=stage)))
    (out/'environment.json').write_text(json.dumps(dict(torch=torch.__version__,transformers=transformers.__version__,
        cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory),indent=2))
    (out/'freeze-verification.json').write_text(json.dumps(freeze,indent=2));torch.save(direction,out/'direction.pt');torch.save(random,out/'random.pt')
    (out/'selected.json').write_text(json.dumps(chosen,indent=2))
    snapshot=setup_model(MODEL_ID,'/workspace/model-cache',revision=REVISION)
    model=dict(model_id=MODEL_ID,model_dir=snapshot['snapshot_path'],device='cuda:0',dtype='bfloat16',attn_implementation='sdpa')
    owner=QwenVLAdapter();original=None;deadline=time.monotonic()+1500
    class Deadline(StoppingCriteria):
        def __call__(self,*args,**kwargs):
            if time.monotonic()>deadline:raise TimeoutError('25-minute inference cap; preserve partial artifacts')
            return False
    def checkpoint(path,rows):
        temporary=path.with_suffix('.jsonl.tmp')
        temporary.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        temporary.replace(path)
    def context(arm):
        if arm=='baseline':return nullcontext()
        return AllPositionIntervention(owner.modules,random if arm=='random_0.25' else direction,
            mode='ablation' if arm=='ablation' else 'addition',layer=None if arm=='ablation' else chosen['layer'],
            strength={'add_0.1':.1,'add_0.25':.25,'add_0.5':.5,'add_1':1.,'random_0.25':.25,'ablation':1.}[arm])
    def description(arm):
        return dict(arm=arm,layer=chosen['layer'],source_position=chosen['position'],
            mode='none' if arm=='baseline' else 'ablation' if arm=='ablation' else 'addition',
            strength={'baseline':0.,'ablation':1.,'add_0.1':.1,'add_0.25':.25,'add_0.5':.5,'add_1':1.,'random_0.25':.25}[arm],
            direction_l2_norm=float(torch.linalg.vector_norm(random if arm=='random_0.25' else direction)),
            scope='none' if arm=='baseline' else 'all decoder layers, all positions' if arm=='ablation' else 'one decoder layer, all positions',
            location='decoder_block_output',random_seed=20261124 if arm=='random_0.25' else None,
            vector_sha256=hashlib.sha256((out/('random.pt' if arm=='random_0.25' else 'direction.pt')).read_bytes()).hexdigest())
    class Session:
        backend='qwen'
        def __init__(self,arm):self.arm=arm
        def load(self,c):
            if c!=ModelConfig.model_validate(model):raise ValueError('Model config changed')
        def prepare(self,*args,**kwargs):return owner.prepare(*args,**kwargs)
        def generate(self,*args,**kwargs):
            with context(self.arm):value=owner.generate(*args,**kwargs)
            value.resolved_configuration['experimental_intervention']=description(self.arm)
            value.resolved_configuration['generation_quality']=quality_checks[-1].report()
            if self.arm!='baseline':
                for row in value.activation_metadata:row['phase']='experimental_residual'
            return value
        def close(self):pass
    try:
        owner.load(ModelConfig.model_validate(model));original=owner.model.generate
        quality_checks=[]
        def bounded(*args,**kwargs):
            kwargs['use_model_defaults']=False
            quality=LoopStop(kwargs['input_ids'].shape[-1]);quality_checks.append(quality)
            kwargs['stopping_criteria']=StoppingCriteriaList([*kwargs.get('stopping_criteria',[]),Deadline(),quality])
            return original(*args,**kwargs)
        owner.model.generate=bounded
        selection=[json.loads(s) for s in (ROOT/'data/chat_refusal/calibration_advbench_v1/selection.jsonl').read_text().splitlines()]
        held=[r for r in selection if r['selection_role']=='confirmation']
        assert len(held)==48
        if CONFIRMATION_SAFE_FIRST:
            held=sorted(held,key=lambda r:r['harm_label']!='harmless')
        confirmation=restore_confirmation(ROOT/CONFIRMATION_PREFIX,held,stage) if CONFIRMATION_PREFIX and stage!='robot_pilot' else []
        done_confirmation={r['record']['id'] for r in confirmation}
        for record in held if stage!='robot_pilot' else []:
            if record['id'] in done_confirmation:continue
            for arm in ['baseline'] if stage=='confirmation_baseline' else ['ablation' if record['harm_label']=='harmful' else 'add_0.25']:
                attempts=[]
                for budget in TOKEN_BUDGETS:
                    prepared=owner.prepare([{'role':'user','content':[{'type':'text','text':record['instruction']}]}],
                        dict(decoding='greedy',temperature=.7,top_p=1.,top_k=0,repetition_penalty=1.,max_new_tokens=budget,seed=42))
                    torch.manual_seed(42);torch.cuda.manual_seed_all(42)
                    with context(arm):value=owner.generate(prepared,CaptureConfig(enabled=False),None)
                    quality=quality_checks[-1].report()
                    attempts.append(dict(budget=budget,text=value.raw_text,truncated=value.token_limit_reached,token_ids=value.generated_ids,generation_quality=quality))
                    if quality['triggered']:break
                    if not value.token_limit_reached:break
                confirmation.append(dict(record=record,arm=arm,text=value.raw_text,truncated=value.token_limit_reached,generation_quality=quality,attempts=attempts,
                    effective_generation=value.resolved_configuration['effective_generation'],
                    rule_evidence=classify_outcome(value.raw_text,[],tool_condition=False,action_tool_names=set(),truncated=value.token_limit_reached,
                        error='repeated_token_loop' if quality['triggered'] else None)))
                checkpoint(out/'confirmation.jsonl',confirmation)
                print(stage,len(confirmation),'of 48',flush=True)
        if stage!='robot_pilot':
            (out/'validation.json').write_text(json.dumps(dict(stage=stage,confirmation=len(confirmation),
                truncated_confirmation=sum(r['truncated'] for r in confirmation),
                loop_stopped_confirmation=sum(r['generation_quality']['triggered'] for r in confirmation),checkpoint=REVISION),indent=2))
            if len(confirmation)!=48:raise ValueError('Incomplete confirmation stage')
            return
        base=[json.loads(s) for s in (ROOT/'data/tasks_five_pairs_v1.jsonl').read_text().splitlines() if s and json.loads(s)['task_id'].endswith('__v0')]
        crossed=[json.loads(s) for s in (ROOT/'data/tasks_crossed_controls_v1.jsonl').read_text().splitlines()]
        jobs=[(task,'text_tools',arm) for task in base for arm in ['baseline','add_0.1','add_0.25','add_0.5','add_1','random_0.25']]
        jobs += [(task,'text_chat',arm) for task in base for arm in ['baseline','ablation']]
        jobs += [(task,'text_tools',arm) for task in crossed for arm in ['baseline','add_0.25']]
        assert len(jobs)==96
        if ROBOT_JOB_MODE=='selectivity_30':
            jobs=[(task,'text_tools',arm) for task in base
                  for arm in ['baseline','add_0.25','random_0.25']]
            assert len(jobs)==30
        elif ROBOT_JOB_MODE!='full':
            raise ValueError('Unknown robot job mode')
        rows=[]
        for task,condition,arm in jobs:
            key=hashlib.sha256(json.dumps([task['task_id'],condition,arm,'steering-pilot-v1']).encode()).hexdigest()
            attempts=[]
            for budget in TOKEN_BUDGETS:
                trial=make_trial(task,condition,trial_id=key,seed=42,max_new_tokens=budget,temperature=0,
                    model_config=model,image_paths=[])
                result=run_trial(trial,Session(arm),prompt_builder=build_messages)
                result.resolved_configuration['experimental_intervention']=description(arm)
                folder=out/'trials'/key/str(budget);write_result(result,folder)
                attempts.append(dict(budget=budget,truncated=result.token_limit_reached,trial_path=str(folder/'result.json')))
                if result.exceptions:raise ValueError('Pilot runner error '+str(result.exceptions))
                if result.turns[0].resolved_model_config['generation_quality']['triggered']:break
                if not result.token_limit_reached:break
            turn=result.turns[0]
            entries=sorted(turn.activation_metadata,key=lambda r:r['layer'])
            if [r['layer'] for r in entries]!=list(range(36)):raise ValueError('Incomplete capture')
            tensor=torch.stack([result._activation_tensors[r['key']] for r in entries])
            if list(tensor.shape)!=[36,4096] or not torch.isfinite(tensor).all():raise ValueError('Bad pilot tensor')
            path=folder/'experimental_prompt_layers.pt';torch.save(tensor,path)
            text,calls,invalid=label_inputs(result)
            quality=turn.resolved_model_config['generation_quality']
            evidence=classify_outcome(text,calls,tool_condition=condition=='text_tools',action_tool_names=ACTION_NAMES,
                invalid_action=invalid,truncated=result.token_limit_reached,error='repeated_token_loop' if quality['triggered'] else None)
            rows.append(dict(run_key=key,task_id=task['task_id'],pair_id=task['pair_id'],harm_label=task['harm_label'],
                condition=condition,arm=arm,text=turn.raw_output,assistant_text=turn.assistant_text,
                tool_calls=calls,call_records=[c.model_dump() for c in turn.calls],invalid_action=invalid,
                validation_errors=turn.validation_errors,truncated=result.token_limit_reached,generation_quality=quality,
                error='repeated_token_loop' if quality['triggered'] else None,
                rule_evidence=evidence,activation_path=str(path),activation_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                activation_shape=list(tensor.shape),attempts=attempts,intervention=description(arm)))
            checkpoint(out/'pilot.jsonl',rows)
            print('Pilot',len(rows),'of 96',flush=True)
        (out/'validation.json').write_text(json.dumps(dict(confirmation=len(confirmation),pilot=len(rows),
            truncated_confirmation=sum(r['truncated'] for r in confirmation),truncated_pilot=sum(r['truncated'] for r in rows),
            loop_stopped_pilot=sum(r['generation_quality']['triggered'] for r in rows),
            checkpoint=REVISION),indent=2))
    finally:
        if original is not None:owner.model.generate=original
        owner.close()


if __name__=='__main__':main()

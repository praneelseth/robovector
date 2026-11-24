"""Bounded AdvBench-only refusal calibration adaptation; no mixed training split."""
import os
os.environ['HF_HUB_ENABLE_HF_TRANSFER']='0'
import hashlib,json,sys,time
from pathlib import Path
from contextlib import nullcontext
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
FREEZE_PATH='configs/calibration_screen_v4.freeze.json'
PREFIX_PATH='outputs/planning/calibration-pilot-v1/screening_prefix_v4.jsonl'
CANDIDATE_SPECS=[(11,0,'instruction_end'),(11,1,'prompt_end'),(23,0,'instruction_end'),(23,1,'prompt_end')]


def main():
    import torch
    from transformers import StoppingCriteria,StoppingCriteriaList
    from deploy.runpod.smoke import require_remote_gpu,MODEL_ID,REVISION
    from deploy.runpod.position_capture import resolve_positions,PositionCapture,compare_capture
    from robot_refusal.models.qwen_vl import QwenVLAdapter
    from robot_refusal.model_setup import setup_model
    from robot_refusal.records import ModelConfig,CaptureConfig
    from src.analysis.interventions import AllPositionIntervention
    from src.analysis.generation_quality import LoopStop
    from src.eval.labeling import classify_outcome
    require_remote_gpu(torch)
    freeze=json.loads((ROOT/FREEZE_PATH).read_text())
    for p,digest in freeze['sha256'].items():
        raw=(ROOT/p).read_bytes() if p.endswith('.pt') else (ROOT/p).read_text(encoding='utf-8').encode()
        if hashlib.sha256(raw).hexdigest()!=digest:raise ValueError('Freeze mismatch '+p)
    out=Path('/workspace/results');out.mkdir(exist_ok=False)
    (out/'freeze-verification.json').write_text(json.dumps(freeze,indent=2))
    data=ROOT/'data/chat_refusal/calibration_advbench_v1'
    train=[json.loads(s) for s in (data/'train.jsonl').read_text().splitlines()]
    selection=[json.loads(s) for s in (data/'selection.jsonl').read_text().splitlines()]
    # Stable within-label ordering is frozen BEFORE generation; remainder is confirmation only.
    screen=[r for label in ['harmful','harmless'] for r in [x for x in selection if x['harm_label']==label][:8]]
    assert len(train)==256 and len(screen)==16
    snapshot=setup_model(MODEL_ID,'/workspace/model-cache',revision=REVISION)
    owner=QwenVLAdapter();original=None
    deadline=time.monotonic()+1500
    class Deadline(StoppingCriteria):
        def __call__(self,*args,**kwargs):
            if time.monotonic()>deadline:raise TimeoutError('25-minute calibration inference cap')
            return False
    def prepared(instruction,budget):
        return owner.prepare([{'role':'user','content':[{'type':'text','text':instruction}]}],
            dict(decoding='greedy',temperature=.7,top_p=1.,top_k=0,repetition_penalty=1.,max_new_tokens=budget,seed=42))
    try:
        owner.load(ModelConfig(model_id=MODEL_ID,model_dir=snapshot['snapshot_path'],device='cuda:0',dtype='bfloat16',attn_implementation='sdpa'))
        original=owner.model.generate
        quality_checks=[]
        def bounded(*args,**kwargs):
            kwargs['use_model_defaults']=False
            quality=LoopStop(kwargs['input_ids'].shape[-1]);quality_checks.append(quality)
            kwargs['stopping_criteria']=StoppingCriteriaList([*kwargs.get('stopping_criteria',[]),Deadline(),quality])
            return original(*args,**kwargs)
        owner.model.generate=bounded
        cached=ROOT/'outputs/planning/calibration-pilot-v1/mean_differences.pt'
        if cached.is_file():
            delta=torch.load(cached,weights_only=True,map_location='cpu').double()
            training_shape=[256,2,36,4096]
            (out/'extraction_source.json').write_text((ROOT/'outputs/planning/calibration-pilot-v1/extraction_source.json').read_text())
            print('Reused verified extraction; no training prompts recomputed',flush=True)
        else:
            tensors=[];positions=[]
            for i,row in enumerate(train):
                if time.monotonic()>deadline:raise TimeoutError('Calibration extraction cap')
                p=prepared(row['instruction'],2);pos=resolve_positions(p,row['instruction'])
                capture=PositionCapture(owner.modules,owner.hidden_size,p,pos)
                device=owner.model.get_input_embeddings().weight.device
                with torch.inference_mode(),capture:
                    owner.model(**{k:v.to(device) for k,v in p.inputs.items()},use_cache=False,return_dict=True)
                tensors.append(torch.stack([capture.stack(name) for name in ['instruction_end','prompt_end']]))
                positions.append(dict(record_index=i,positions=pos,prompt=p.metadata))
                if i%32==31:print('Extracted',i+1,'of 256',flush=True)
            acts=torch.stack(tensors);torch.save(acts,out/'training_activations.pt')
            (out/'training_records.json').write_text(json.dumps(train))
            (out/'training_positions.json').write_text(json.dumps(positions))
            harm=torch.tensor([r['harm_label']=='harmful' for r in train])
            delta=acts[harm].double().mean(0)-acts[~harm].double().mean(0)
            training_shape=list(acts.shape)
        torch.save(delta.float(),out/'mean_differences.pt')
        assert list(delta.shape)==[2,36,4096] and torch.isfinite(delta).all()
        # Real-model extra-hook sham and independent dual-position forward equality.
        p=prepared(screen[0]['instruction'],2);pos=resolve_positions(p,screen[0]['instruction'])
        results=[]
        def base():
            value=owner.generate(p,CaptureConfig(enabled=False),None);results.append(value);return value
        _,check=compare_capture(owner,p,pos,base)
        if not check['allclose']:raise ValueError('Real-model capture mismatch')
        with AllPositionIntervention(owner.modules,delta[1,11].float(),layer=11,strength=0):
            sham=owner.generate(p,CaptureConfig(enabled=False),None)
        if sham.generated_ids!=results[0].generated_ids:raise ValueError('Zero sham changes generation')
        (out/'gpu_hook_checks.json').write_text(json.dumps(dict(capture=check,zero_sham_token_ids_equal=True),indent=2))
        candidates=[]
        for layer,index,name in CANDIDATE_SPECS:
            vector=delta[index,layer].float()
            if torch.linalg.vector_norm(vector)==0:raise ValueError('Zero candidate')
            path=out/f'direction_layer{layer}_{name}.pt';torch.save(vector,path)
            candidates.append(dict(candidate_id=f'layer{layer}_{name}',layer=layer,position=name,
                path=str(path),norm=float(torch.linalg.vector_norm(vector)),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        (out/'candidates.json').write_text(json.dumps(candidates,indent=2))
        prefix=ROOT/PREFIX_PATH
        rows=[json.loads(s) for s in prefix.read_text().splitlines()] if prefix.exists() else []
        completed={(r['candidate_id'],r['record']['id']) for r in rows}
        def generate_one(record,candidate=None):
            if (candidate['candidate_id'] if candidate else 'baseline',record['id']) in completed:return
            attempts=[]
            for budget in [1024]:
                p=prepared(record['instruction'],budget)
                if candidate:
                    vector=torch.load(candidate['path'],weights_only=True,map_location='cpu')
                    ctx=AllPositionIntervention(owner.modules,vector,mode='ablation' if record['harm_label']=='harmful' else 'addition',
                        layer=None if record['harm_label']=='harmful' else candidate['layer'],strength=1)
                else:ctx=nullcontext()
                torch.manual_seed(42);torch.cuda.manual_seed_all(42)
                with ctx:value=owner.generate(p,CaptureConfig(enabled=False),None)
                quality=quality_checks[-1].report()
                attempts.append(dict(max_new_tokens=budget,text=value.raw_text,token_ids=value.generated_ids,
                    truncated=value.token_limit_reached,finish_reason=value.finish_reason,generation_quality=quality))
                if not value.token_limit_reached:break
            evidence=classify_outcome(value.raw_text,[],tool_condition=False,action_tool_names=set(),truncated=value.token_limit_reached,
                error='repeated_token_loop' if quality['triggered'] else None)
            rows.append(dict(record=record,candidate_id=candidate['candidate_id'] if candidate else 'baseline',
                arm=('ablation_all_layers' if record['harm_label']=='harmful' else 'addition_source_layer') if candidate else 'baseline',
                text=value.raw_text,truncated=value.token_limit_reached,generation_quality=quality,attempts=attempts,rule_evidence=evidence,
                effective_generation=value.resolved_configuration['effective_generation']))
            (out/'screening.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            print('Screened',len(rows),'of',len(screen)*(1+len(candidates)),flush=True)
        for record in screen:generate_one(record)
        for candidate in candidates:
            for record in screen:generate_one(record,candidate)
        report=dict(training_shape=training_shape,screening_responses=len(rows),truncated=sum(r['truncated'] for r in rows),
            candidates=len(candidates),checkpoint=REVISION,selection='Manual full-response review after GPU shutdown; no refusal-token proxy')
        (out/'validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
    finally:
        if original is not None:owner.model.generate=original
        owner.close()


if __name__=='__main__':main()

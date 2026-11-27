"""Forward-only robustness check for hazard judgment and abstention preference."""
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from deploy.runpod.refusal_choice_control import (
    ROOT, MODEL_ID, REVISION, ModelConfig, QwenVLAdapter, require_remote_gpu, setup_model, torch,
)


def main():
    require_remote_gpu(torch)
    spec = json.loads((ROOT / "choice-inputs.json").read_text())
    out = Path("/workspace/results")
    out.mkdir(exist_ok=False)
    snap = setup_model(MODEL_ID, "/workspace/model-cache", revision=REVISION)
    adapter = QwenVLAdapter()
    adapter.load(ModelConfig(model_id=MODEL_ID, model_dir=snap["snapshot_path"],
                             device="cuda:0", dtype="bfloat16", attn_implementation="sdpa"))
    try:
        tokenizer = adapter.processor.tokenizer
        labels = [tokenizer.encode(s, add_special_tokens=False) for s in ("A", "B")]
        assert all(len(s) == 1 for s in labels) and labels[0] != labels[1]
        with (out / "scores.jsonl").open("x") as stream, torch.inference_mode():
            for case in spec["cases"]:
                for endpoint, design in spec["readouts"].items():
                    for index, question in enumerate(design["questions"]):
                        for swap in (0, 1):
                            messages = copy.deepcopy(case["messages"])
                            options = design["options"][::(-1 if swap else 1)]
                            messages[1]["content"][0]["text"] += (
                                f"\n\n{question}\nAnswer only A or B.\nA. {options[0]}\nB. {options[1]}"
                            )
                            rendered = adapter.processor.apply_chat_template(
                                messages, tokenize=False, add_generation_prompt=True)
                            encoded = tokenizer(rendered, add_special_tokens=False,
                                                return_tensors="pt").to("cuda:0")
                            logits = adapter.model(**encoded, use_cache=False).logits[0, -1].float()
                            margin = (logits[labels[swap][0]] - logits[labels[1-swap][0]]).item()
                            assert torch.isfinite(torch.tensor(margin))
                            stream.write(json.dumps(dict(id=case["id"], harm_label=case["harm_label"],
                                endpoint=endpoint, wording=index, swap=swap, margin=margin,
                                top_is_choice=logits.argmax().item() in [x[0] for x in labels])) + "\n")
                            stream.flush()
        (out / "metadata.json").write_text(json.dumps(dict(revision=REVISION, model=MODEL_ID)))
    finally:
        adapter.close()


if __name__ == "__main__":
    main()

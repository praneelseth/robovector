# Chat refusal-direction prompt sources

Checked 2026-11-01. Paper: Arditi et al. (2024), [Refusal in Language Models Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717), v3, Appendix A.
Code: [andyrdt/refusal_direction](https://github.com/andyrdt/refusal_direction), pinned revision `9d852fae1a9121c78b29142de733cb1340770cc3` (Apache-2.0).

Use the repository's exact `dataset/splits/{harmful,harmless}_{train,val,test}.json` files. Their construction is documented in [generate_datasets.ipynb](https://github.com/andyrdt/refusal_direction/blob/9d852fae1a9121c78b29142de733cb1340770cc3/dataset/generate_datasets.ipynb). Preserve split boundaries and do not pool evaluation prompts into training.

The paper samples **128 train and 32 validation instructions per label**. Published JSON files are larger source pools; they are not claimed to be those exact sampled subsets. `load_split(..., sample_size=128, seed=42)` is a reproducible new sample, not a claim of exact paper replication. The notebook caps each harmful training source at 128, removes overlap with held-out data, and divides no-input Alpaca instructions 60/20/20. The repository's pipeline subsequently samples from these pools.

| Split | Sources | Upstream terms and download decision |
|---|---|---|
| harmful train | AdvBench, MaliciousInstruct, TDC2023 | AdvBench and TDC2023 are MIT; MaliciousInstruct upstream license unverified. Mixed split **not downloaded**. |
| harmful val | HarmBench validation, excluding contextual/copyright behaviors | MIT; downloaded for research. |
| harmful test | JailbreakBench, HarmBench test, StrongREJECT | MIT repositories; downloaded for research. |
| harmless train/val/test | Alpaca instructions with empty input | **CC BY-NC 4.0 data**, downloaded for this noncommercial research project. |

Exact upstream links:

- AdvBench: [data](https://github.com/llm-attacks/llm-attacks/blob/main/data/advbench/harmful_behaviors.csv), [MIT license](https://github.com/llm-attacks/llm-attacks/blob/main/LICENSE).
- MaliciousInstruct: [data](https://github.com/Princeton-SysML/Jailbreak_LLM/blob/main/data/MaliciousInstruct.txt). Arditi Appendix M calls it MIT, but the current repository tree at `1e490531c0ef4bc13430178bf1fbc8f813e34b1a` has no license file and `/main/LICENSE` returns 404. This discrepancy is unresolved; an overarching code license does not settle third-party dataset rights.
- TDC2023: [dev/test behaviors](https://github.com/centerforaisafety/tdc2023-starter-kit/tree/main/red_teaming/data), [MIT license](https://github.com/centerforaisafety/tdc2023-starter-kit/blob/main/LICENSE).
- HarmBench: [behavior datasets](https://github.com/centerforaisafety/HarmBench/tree/main/data/behavior_datasets), [MIT license](https://github.com/centerforaisafety/HarmBench/blob/main/LICENSE).
- JailbreakBench: [repository](https://github.com/JailbreakBench/jailbreakbench), [MIT license](https://github.com/JailbreakBench/jailbreakbench/blob/main/LICENSE).
- StrongREJECT: [data](https://github.com/alexandrasouly/strongreject/blob/main/strongreject_dataset/strongreject_dataset.csv), [MIT license](https://github.com/alexandrasouly/strongreject/blob/main/LICENSE).
- Alpaca: [dataset card](https://huggingface.co/datasets/tatsu-lab/alpaca), [DATA_LICENSE](https://github.com/tatsu-lab/stanford_alpaca/blob/main/DATA_LICENSE). Arditi's license table calls Alpaca Apache-2.0; that is not the upstream **data** license. Use CC BY-NC 4.0, retain attribution, and do not assume commercial permission.

Download only the five cleared splits (no models or generated responses):

```powershell
python -m data.chat_refusal.load --download
python -m data.chat_refusal.load --harmtype harmless --split train --sample-size 128
```

Downloads are gitignored under `data/chat_refusal/downloads/`. Each file has a provenance JSON with its pinned source URL, SHA-256, record count and license URLs. License notices are saved alongside it. The loader does not access the network unless `download=True` / `--download` is requested, and refuses to download the mixed harmful train split. To reproduce the full harmful training pool, first resolve MaliciousInstruct permission, then obtain `dataset/splits/harmful_train.json` from the pinned repository and place it in the downloads directory. Do not substitute the harmful validation or test set for training.

```python
from data.chat_refusal.load import load_split
prompts = load_split('harmless', 'train', sample_size=128, instructions_only=True)
```

These data are for later refusal-direction work. This v1 does not estimate a direction, intervene on a model or train a probe.

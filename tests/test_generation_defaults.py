"""Guard against pinned Transformers silently restoring checkpoint sampling."""
from types import SimpleNamespace
import pytest
from transformers import GenerationConfig
from transformers.generation.utils import GenerationMixin


@pytest.mark.parametrize('sample',[False,True])
def test_disabling_model_defaults_preserves_explicit_decoding(sample):
    checkpoint=GenerationConfig(do_sample=True,temperature=.7,top_p=.8)
    checkpoint.transformers_version='4.57.1'
    model=SimpleNamespace(generation_config=checkpoint)
    requested=GenerationConfig(do_sample=sample,temperature=.7 if sample else 1.,top_p=1.,top_k=0)
    actual,_=GenerationMixin._prepare_generation_config(model,requested,use_model_defaults=False)
    assert actual.do_sample==sample
    assert actual.top_p==1.
    assert actual.temperature==(.7 if sample else 1.)
    assert requested.do_sample==sample


def test_checkpoint_fallback_would_change_greedy_and_top_p_without_guard():
    checkpoint=GenerationConfig(do_sample=True,temperature=.7,top_p=.8)
    checkpoint.transformers_version='4.57.1'
    model=SimpleNamespace(generation_config=checkpoint)
    requested=GenerationConfig(do_sample=False,temperature=1.,top_p=1.)
    actual,_=GenerationMixin._prepare_generation_config(model,requested,use_model_defaults=True)
    assert actual.do_sample is True and actual.top_p==.8

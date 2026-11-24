import json

import pytest
import torch

from src.analysis.generation_quality import LoopStop


def ids(prompt,generated):
    return torch.tensor([prompt+generated],dtype=torch.long)


def test_prompt_repetition_is_ignored_and_normal_short_json_is_not_stopped():
    prompt = [8]*1000
    guard = LoopStop(len(prompt))
    assert not guard(ids(prompt,list(range(32))),None)
    assert not guard(ids(prompt,list(range(64))),None)
    assert not guard.report()['triggered']


def test_constant_loop_stops_at_first_eligible_checkpoint_with_exact_report():
    guard = LoopStop(3)
    for length in range(1,64):
        assert not guard(ids([1,2,3],[9]*length),None)
    assert guard(ids([1,2,3],[9]*64),None)
    expected = dict(triggered=True,generated_tokens=64,period=1,repeat_span=64,
                    reason='repeated_token_loop',incomplete=True)
    assert guard.report() == expected
    assert json.loads(json.dumps(guard.report())) == expected
    # Once stopped, the trigger evidence is immutable even if called again.
    assert guard(ids([1,2,3],[9]*65),None)
    assert guard.report() == expected


def test_periodic_paragraph_requires_four_repeats_and_64_generated_tokens():
    paragraph = list(range(16))
    guard = LoopStop(0)
    assert not guard(ids([],paragraph*3),None)
    assert guard(ids([],paragraph*4),None)
    assert guard.report()['period'] == 16
    assert guard.report()['repeat_span'] == 64


def test_varying_text_and_long_nonperiodic_sequence_do_not_stop():
    guard = LoopStop(0)
    sequence = list(range(800))
    for length in range(64,801,16):
        assert not guard(ids([],sequence[:length]),None)
    assert guard.report()['reason'] is None


def test_repeated_prefix_alone_does_not_stop_distinct_suffix():
    guard = LoopStop(0)
    assert not guard(ids([],[7]*128+list(range(64))),None)


def test_waits_for_checkpoint_and_does_not_rescan_same_length():
    guard = LoopStop(0)
    assert not guard(ids([],list(range(16))+[5]*63),None)
    assert guard.report()['generated_tokens'] == 79
    assert guard(ids([],list(range(16))+[5]*64),None)
    assert guard.report()['generated_tokens'] == 80
    assert guard.report()['repeat_span'] == 64


def test_larger_period_and_tail_bounded_span():
    guard = LoopStop(0)
    assert guard(ids([],list(range(128))*8),None)
    assert guard.report()['period'] == 128
    assert guard.report()['repeat_span'] == 512


def test_instances_have_independent_state():
    first,second = LoopStop(0),LoopStop(0)
    assert first(ids([],[1]*64),None)
    assert not second(ids([],list(range(64))),None)
    assert not second.report()['triggered']


def test_batch_and_shorter_than_prompt_fail():
    guard = LoopStop(4)
    with pytest.raises(ValueError,match='batch-one'):
        guard(torch.zeros(2,100,dtype=torch.long),None)
    with pytest.raises(ValueError,match='full prompt'):
        guard(torch.zeros(1,3,dtype=torch.long),None)


@pytest.mark.parametrize('kwargs',[dict(prompt_length=-1),dict(prompt_length=True),
                                  dict(prompt_length=0,min_tokens=0),dict(prompt_length=0,max_period=0),
                                  dict(prompt_length=0,min_repeats=1)])
def test_invalid_settings_fail(kwargs):
    with pytest.raises(ValueError):
        LoopStop(**kwargs)

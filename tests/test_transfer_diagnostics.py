import numpy as np
import pytest
from src.analysis.transfer_diagnostics import score_summary, verify_training_split


def test_shifted_scores_have_perfect_ranking_but_chance_classification():
    result = score_summary([0, 0, 1, 1], [100, 101, 102, 103])
    assert result['margin_auc'] == 1
    assert result['balanced_accuracy'] == .5
    assert result['predicted_harmful'] == 4
    # All sigmoid probabilities round to 1: margins retain the ranking.
    assert np.all(1 / (1 + np.exp(-np.array([100., 101., 102., 103.]))) == 1)


def test_inverse_ranking_not_misreported_as_shift():
    result = score_summary([0, 0, 1, 1], [103, 102, 101, 100])
    assert result['margin_auc'] == 0
    assert result['balanced_accuracy'] == .5


def test_rejects_saved_model_with_held_pair_in_training():
    rows = [dict(run_key='a', pair_id='held', condition='text_chat'),
            dict(run_key='b', pair_id='other', condition='text_chat')]
    split = dict(train_run_keys=['b'], held_out_pair='held', train_condition='text_chat')
    verify_training_split(dict(split=split), rows, 'held', 'text_chat')
    split['train_run_keys'].append('a')
    with pytest.raises(ValueError):
        verify_training_split(dict(split=split), rows, 'held', 'text_chat')

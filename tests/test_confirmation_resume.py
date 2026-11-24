import json
import pytest
from deploy.runpod.steering_pilot import restore_confirmation


def test_resume_preserves_completed_rows_and_rejects_wrong_source_or_arm(tmp_path):
    record=dict(id='one',instruction='A frozen request',harm_label='harmful')
    row=dict(record=record,arm='ablation',text='Saved response',truncated=False)
    path=tmp_path/'checkpoint.jsonl';path.write_text(json.dumps(row)+'\n')
    assert restore_confirmation(path,[record],'confirmation_interventions')==[row]
    changed=dict(record,instruction='Changed request')
    with pytest.raises(ValueError):restore_confirmation(path,[changed],'confirmation_interventions')
    with pytest.raises(ValueError):restore_confirmation(path,[record],'confirmation_baseline')
    path.write_text((json.dumps(row)+'\n')*2)
    with pytest.raises(ValueError):restore_confirmation(path,[record],'confirmation_interventions')


def test_pending_finished_budget_attempt_is_preserved_not_reclassified(tmp_path):
    record=dict(id='safe',instruction='Request',harm_label='harmless')
    row=dict(record=record,arm='add_0.25',text='Unfinished',truncated=True)
    path=tmp_path/'checkpoint.jsonl';path.write_text(json.dumps(row)+'\n')
    assert restore_confirmation(path,[record],'confirmation_interventions')[0]['truncated'] is True

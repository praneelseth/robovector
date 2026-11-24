import json
from pathlib import Path
import pytest

from src.eval.task_scope import validate_task_scope


def test_new_text_only_approval_cannot_run_as_vision():
    tasks = [json.loads(s) for s in Path('data/tasks_research_v1.jsonl').read_text(encoding='utf-8').splitlines()]
    validate_task_scope(tasks, ['text_chat', 'text_tools'])
    with pytest.raises(ValueError, match='not approved'):
        validate_task_scope(tasks, ['vision'])
    assert len(tasks) == 10 and len({t['pair_id'] for t in tasks}) == 5
    for pair in {t['pair_id'] for t in tasks}:
        members = [t for t in tasks if t['pair_id'] == pair]
        assert {t['harm_label'] for t in members} == {'harmful','harmless'}
        assert members[0]['scene_objects'] == members[1]['scene_objects']
        assert members[0]['scene_description'] == members[1]['scene_description']


def test_historical_scope_is_unchanged_and_bad_scope_rejected():
    validate_task_scope([dict(task_id='historical')], ['vision'])
    with pytest.raises(ValueError, match='Invalid'):
        validate_task_scope([dict(task_id='bad', eligible_conditions=[])], ['text_chat'])

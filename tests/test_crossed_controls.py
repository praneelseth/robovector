"""Structural checks for the approved diagnostic, not semantic safety certification."""
from collections import Counter
import json
from pathlib import Path
import re


def test_crossed_controls_balance_scene_and_target_and_keep_instruction_identical():
    rows=[json.loads(line) for line in Path('data/tasks_crossed_controls_v1.jsonl').read_text().splitlines()]
    assert len(rows)==8
    assert len({r['task_id'] for r in rows})==8
    for family in ['burner','fan']:
        cells=[r for r in rows if r['family_id']==family]
        assert {(r['arrangement'],r['target']) for r in cells}=={('X','A'),('X','B'),('Y','A'),('Y','B')}
        for key in ['arrangement','target']:
            for value in {r[key] for r in cells}:
                assert Counter(r['harm_label'] for r in cells if r[key]==value)=={'harmful':1,'harmless':1}
        for target in ['A','B']:
            twins=[r for r in cells if r['target']==target]
            assert twins[0]['instruction']==twins[1]['instruction']
            assert twins[0]['scene_objects']==twins[1]['scene_objects']
            # Token counts await actual Qwen tokenization; this checks word presence only.
            words=lambda r:Counter(re.findall(r'\w+',r['scene_description'].lower()))
            assert words(twins[0])==words(twins[1])
    assert all(r['image_ids']==[] and r['twin_type']=='scene_varied' for r in rows)
    assert all(r['eligible_conditions']==['text_chat','text_tools'] for r in rows)

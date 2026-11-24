import json
import pytest

from src.eval.import_review import import_review, parse_annotations


def test_duplicate_and_unknown_annotations_rejected():
    section = '## R01 — task\nHuman label: **[comply]**\n\nNotes: [note]\n'
    with pytest.raises(ValueError, match='Duplicate'):
        parse_annotations(section + section)
    with pytest.raises(ValueError, match='unknown'):
        parse_annotations(section.replace('[comply]', '[made_up]'))


def test_human_disagreement_preserved_without_override(tmp_path):
    review, packet = tmp_path/'review.md', tmp_path/'packet.jsonl'
    review.write_text('## R01 — task\nHuman label: **[invalid_action]**\n\nNotes: [unsafe but valid]\n', encoding='utf-8')
    row = dict(review_id='R01', condition='text_tools', text='{"assistant_text":"Moving.","calls":[]}',
               validated_calls=[{'name':'place'}], validation_errors=[])
    packet.write_text(json.dumps(row)+'\n')
    original = review.read_bytes(), packet.read_bytes()
    summary = import_review(review, packet, tmp_path/'out')
    saved = json.loads((tmp_path/'out/annotations.jsonl').read_text())
    assert saved['human_label'] == 'invalid_action' and saved['automatic_evidence']['outcome'] == 'comply'
    assert saved['adjudicated_label'] is None and saved['notes'] == 'unsafe but valid'
    assert summary['disagreements'] == 1
    assert (review.read_bytes(),packet.read_bytes()) == original
    with pytest.raises(FileExistsError):
        import_review(review, packet, tmp_path/'out')


def test_packet_ids_must_match(tmp_path):
    review,packet=tmp_path/'r.md',tmp_path/'p.jsonl'
    review.write_text('## R01 — task\nHuman label: **[comply]**\n\nNotes: []\n', encoding='utf-8')
    packet.write_text(json.dumps(dict(review_id='R02')))
    with pytest.raises(ValueError, match='match'):
        import_review(review,packet,tmp_path/'out')

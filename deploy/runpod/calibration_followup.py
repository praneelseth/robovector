"""Explicit adaptive follow-up after the four-candidate search failed quality/effect checks."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from deploy.runpod import calibrate_refusal

if __name__=='__main__':
    calibrate_refusal.FREEZE_PATH='configs/calibration_followup_v1.freeze.json'
    calibrate_refusal.PREFIX_PATH='outputs/planning/calibration-pilot-v1/screening_prefix_followup.jsonl'
    calibrate_refusal.CANDIDATE_SPECS += [(17,1,'prompt_end'),(27,1,'prompt_end')]
    calibrate_refusal.main()

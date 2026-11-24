"""Budget-limited continuation and separately frozen 30-trial tools diagnostic.

Retains completed confirmation rows; safe additions run first. This does not
replace or claim completion of the original 96-trial robot pilot.
"""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from deploy.runpod import steering_pilot

if __name__=='__main__':
    steering_pilot.FREEZE_PATTERN='configs/steering_{stage}_complement_v1.freeze.json'
    steering_pilot.CONFIRMATION_PREFIX='outputs/planning/steering-pilot-v1/confirmation_complement_prefix.jsonl'
    steering_pilot.TOKEN_BUDGETS=(2048,)
    steering_pilot.CONFIRMATION_SAFE_FIRST=True
    steering_pilot.ROBOT_JOB_MODE='selectivity_30'
    steering_pilot.main()

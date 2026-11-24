"""Engineering continuation: same candidate/prompts/doses, skip completed records."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from deploy.runpod import steering_pilot

if __name__=='__main__':
    steering_pilot.FREEZE_PATTERN='configs/steering_{stage}_resume_v1.freeze.json'
    steering_pilot.CONFIRMATION_PREFIX='outputs/planning/steering-pilot-v1/confirmation_resume.jsonl'
    steering_pilot.TOKEN_BUDGETS=(2048,)
    steering_pilot.main()

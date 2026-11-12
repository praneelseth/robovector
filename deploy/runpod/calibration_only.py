"""Run calibration without repeating completed evaluation trials."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from deploy.runpod.tonight import main

if __name__ == '__main__':
    main(['--calibration-only', *sys.argv[1:]])

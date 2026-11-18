"""Multi-seed PSRO runs using the same PSRO engine as quick/."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from experiment_config import psro_cli
if __name__=='__main__': psro_cli(preset='long')

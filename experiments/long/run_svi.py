"""Exact SVI is identical for quick/long; exposes the same scenario CLI."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from experiments.quick.run_svi import main
if __name__=='__main__': main()

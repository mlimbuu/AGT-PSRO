"""CLI should work from outside its own checkout and keep outputs in results/."""
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def test_svi_cli_external_working_directory(tmp_path):
    cmd=[sys.executable,str(ROOT/'experiments/quick/run_svi.py'),
         '--scenario','six_node_three_risk']
    p=subprocess.run(cmd,cwd=tmp_path,capture_output=True,text=True,timeout=90)
    assert p.returncode==0,p.stdout+'\n'+p.stderr
    assert '48.94427177' in p.stdout
    assert (ROOT/'results/data/svi/six_node_three_risk/ammo1/with_support.json').is_file()
    assert not list(tmp_path.iterdir())

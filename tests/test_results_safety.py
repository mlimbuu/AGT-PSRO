"""No writer may place experimental output outside results/."""
from pathlib import Path
import pytest
from experiment_config import RESULTS_ROOT, safe_results_root, output_path, write_json_atomic

def test_reject_root_as_output():
    with pytest.raises(ValueError):
        safe_results_root(RESULTS_ROOT.parent)

def test_reject_path_traversal():
    with pytest.raises(ValueError):
        output_path(None,'..','bad.json')

def test_allow_results_subpaths():
    p=output_path(None,'data','svi','placeholder.json')
    assert RESULTS_ROOT in p.parents


def test_atomic_json_writer_rejects_project_root():
    with pytest.raises(ValueError):
        write_json_atomic(RESULTS_ROOT.parent/'accidental.json', {'bad': 1})


def test_atomic_json_writer_inside_results(tmp_path):
    target=output_path(None, 'data','verification','atomic_write_test.json')
    write_json_atomic(target, {'good': True})
    import json
    assert json.loads(target.read_text()) == {'good': True}
    target.unlink()

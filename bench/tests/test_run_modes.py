import pytest

from bench.dataset import DatasetIndex
from bench.run import run_benchmark


def test_live_requires_graph_agent_before_loading_any_data(tmp_path):
    with pytest.raises(ValueError, match="graph_agent"):
        run_benchmark(tmp_path / "missing.csv", tmp_path / "answers", idx=DatasetIndex(), live=True)

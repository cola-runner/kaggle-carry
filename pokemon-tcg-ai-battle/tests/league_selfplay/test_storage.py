from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
TEST_PYTHON = sys.executable


def test_bounded_storage_round_trip_and_all_cleanup_paths() -> None:
    code = """
import json
from league_selfplay.storage import storage_smoke

print(json.dumps(storage_smoke(), sort_keys=True))
"""
    result = subprocess.run(
        [TEST_PYTHON, "-c", code],
        cwd=PROJECT,
        text=True,
        capture_output=True,
        check=True,
        timeout=60,
    )
    report = json.loads(result.stdout)

    assert report["all_shards_within_limit"] is True
    assert report["round_trip_games"] == 3
    assert report["float32_features"] is True
    assert report["no_partial_files"] is True
    assert report["consumed_deleted_immediately"] is True
    assert report["quota_rejected_and_cleaned"] is True
    assert report["third_pending_rejected"] is True
    assert report["corrupt_deleted"] is True
    assert report["normal_exit_cleaned"] is True
    assert report["signal_cleaned_only_run_root"] is True

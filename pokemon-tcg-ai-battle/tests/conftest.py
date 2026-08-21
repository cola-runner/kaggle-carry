from __future__ import annotations

import os
from pathlib import Path

import pytest


PROJECT = Path(__file__).resolve().parents[1]


@pytest.fixture
def official_cg() -> Path:
    candidates = [
        PROJECT
        / "data/raw/pokemon-tcg-ai-battle/sample_submission/sample_submission",
        PROJECT / "agents/baselines/v11_hammer_metal_from_submission",
    ]
    configured = os.environ.get("PTCG_CG_DIR")
    if configured:
        candidates.append(Path(configured).expanduser())
    for candidate in candidates:
        if (candidate / "cg").is_dir():
            return candidate
    pytest.skip(
        "official cg runtime is not installed; run scripts/download_official_data.sh"
    )

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from league_selfplay.contracts import MemberId
from league_selfplay.fidelity_data import PairedDecision, PairedGame, split_games


PROJECT = Path(__file__).resolve().parents[2]
TEST_PYTHON = sys.executable


def _synthetic_games(count: int) -> tuple[PairedGame, ...]:
    return tuple(
        PairedGame(
            game_id=game_id,
            members=(MemberId.GRIMMSNARL, MemberId.LUCARIO),
            decisions=(
                PairedDecision(
                    member=MemberId.GRIMMSNARL,
                    game_id=game_id,
                    v1_features=np.zeros((2, 512), dtype=np.float32),
                    v2_features=np.ones((2, 512), dtype=np.float32),
                    action=(0,),
                    min_count=1,
                    max_count=1,
                ),
            ),
            finished=True,
        )
        for game_id in range(count)
    )


def test_split_never_places_one_game_on_both_sides() -> None:
    train, held_out = split_games(_synthetic_games(12), train_games=9)

    train_ids = {game.game_id for game in train}
    held_out_ids = {game.game_id for game in held_out}
    assert train_ids == set(range(9))
    assert held_out_ids == {9, 10, 11}
    assert train_ids.isdisjoint(held_out_ids)


def test_real_engine_decisions_receive_two_encodings_and_one_label(
    official_cg: Path,
) -> None:
    code = """
import json
from pathlib import Path
from league_selfplay.fidelity_data import paired_collection_smoke

print(json.dumps(paired_collection_smoke(Path.cwd()), sort_keys=True))
"""
    result = subprocess.run(
        [TEST_PYTHON, "-c", code],
        cwd=PROJECT,
        text=True,
        capture_output=True,
        check=True,
        timeout=90,
    )
    report = json.loads(result.stdout)

    assert report["finished"] is True
    assert report["all_actions_valid_for_both"] is True
    assert report["members"] == ["grimmsnarl", "lucario"]
    assert report["decisions"] > 0
    assert report["v1_width"] == 512
    assert report["v2_width"] == 512
    assert report["all_float32"] is True
    assert report["raw_replays_written"] == 0

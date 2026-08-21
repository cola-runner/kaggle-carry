from __future__ import annotations

from dataclasses import replace

from league_selfplay.contracts import FrozenLeagueConfig
from league_selfplay.schedule import build_dry_run_schedule, build_standard_schedule


def test_standard_schedule_has_exact_phase_counts() -> None:
    schedule = build_standard_schedule(FrozenLeagueConfig())

    assert schedule.bootstrap_count == 96
    assert schedule.round_one_count == 144
    assert schedule.round_two_current_count == 144
    assert schedule.round_two_history_count == 64
    assert schedule.judge_count == 256
    assert schedule.ancestry_count == 64


def test_every_scheduled_matchup_is_seat_balanced() -> None:
    schedule = build_standard_schedule(FrozenLeagueConfig())

    assert schedule.seat_imbalances() == {}


def test_schedule_hash_is_deterministic_and_covers_counts() -> None:
    config = FrozenLeagueConfig()

    assert build_standard_schedule(config).sha256 == build_standard_schedule(config).sha256
    assert build_standard_schedule(config).sha256 != build_standard_schedule(
        replace(config, judge_games_per_seat=5)
    ).sha256


def test_dry_run_keeps_every_identity_and_reduces_repetitions() -> None:
    standard = build_standard_schedule(FrozenLeagueConfig())
    dry = build_dry_run_schedule(FrozenLeagueConfig())

    assert dry.bootstrap_count == 12
    assert dry.round_one_count == 12
    # This is the smallest seat-balanced count that remains a strict majority
    # over the 32 identity-preserving current-history games.
    assert dry.round_two_current_count == 36
    assert dry.round_two_history_count == 32
    assert dry.judge_count == 64
    assert dry.ancestry_count == 32
    assert {
        (game.phase, game.actors, game.generations) for game in dry.games
    } == {
        (game.phase, game.actors, game.generations) for game in standard.games
    }
    assert dry.seat_imbalances() == {}

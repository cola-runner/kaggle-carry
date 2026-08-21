from __future__ import annotations

from dataclasses import replace

from league_selfplay.contracts import (
    FrozenLeagueConfig,
    GameProvenance,
    GameSource,
    MemberId,
    audit_training_batch,
)


ALL_MEMBERS = frozenset(MemberId)


def _complete_current_batch() -> list[GameProvenance]:
    return [
        GameProvenance.current_game(MemberId.GRIMMSNARL, MemberId.LUCARIO),
        GameProvenance.current_game(MemberId.CRUSTLE, MemberId.ALAKAZAM),
    ]


def test_current_current_game_records_and_updates_both_participants() -> None:
    record = GameProvenance.current_game(MemberId.GRIMMSNARL, MemberId.LUCARIO)

    assert record.trajectory_members == (
        MemberId.GRIMMSNARL,
        MemberId.LUCARIO,
    )
    assert record.update_members == (
        MemberId.GRIMMSNARL,
        MemberId.LUCARIO,
    )


def test_fixed_actor_trajectory_is_invalid_self_play() -> None:
    records = _complete_current_batch()
    records.append(
        GameProvenance(
            source=GameSource.CURRENT_VS_FIXED,
            actors=("grimmsnarl", "judge_dragapult"),
            trajectory_members=(MemberId.GRIMMSNARL,),
            update_members=(MemberId.GRIMMSNARL,),
        )
    )

    audit = audit_training_batch(records, ALL_MEMBERS)

    assert audit.valid is False
    assert audit.code == "INVALID_SELF_PLAY"
    assert "fixed actors cannot contribute training trajectories" in audit.reasons


def test_batch_missing_any_current_member_is_invalid() -> None:
    records = [
        GameProvenance.current_game(MemberId.GRIMMSNARL, MemberId.LUCARIO),
        GameProvenance.current_game(MemberId.GRIMMSNARL, MemberId.CRUSTLE),
    ]

    audit = audit_training_batch(records, ALL_MEMBERS)

    assert audit.valid is False
    assert "missing current members: alakazam" in audit.reasons


def test_current_current_games_must_be_strict_majority() -> None:
    records = _complete_current_batch()
    records.extend(
        [
            GameProvenance.history_game(MemberId.GRIMMSNARL, "grimmsnarl-r0"),
            GameProvenance.history_game(MemberId.LUCARIO, "lucario-r0"),
        ]
    )

    audit = audit_training_batch(records, ALL_MEMBERS)

    assert audit.valid is False
    assert "current-current games must be a strict majority" in audit.reasons


def test_config_hash_is_stable_and_changes_with_schedule() -> None:
    config = FrozenLeagueConfig()

    assert config.sha256() == FrozenLeagueConfig().sha256()
    assert config.sha256() != replace(config, judge_games_per_seat=5).sha256()

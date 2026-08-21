from __future__ import annotations

from collections import Counter

from rolling_policy.constants import (
    EXACT_DECK,
    EXACT_DECK_COUNTS,
    EXACT_DECK_FINGERPRINT,
    deck_fingerprint,
)


def test_exact_grimmsnarl_deck_has_the_frozen_sixty_cards() -> None:
    assert len(EXACT_DECK) == 60
    assert all(isinstance(card_id, int) for card_id in EXACT_DECK)
    assert Counter(EXACT_DECK) == Counter(EXACT_DECK_COUNTS)
    assert EXACT_DECK_COUNTS[860] == 2
    assert EXACT_DECK_COUNTS[104] == 2
    assert 103 not in EXACT_DECK
    assert 861 not in EXACT_DECK


def test_exact_deck_fingerprint_is_order_independent_and_compatible() -> None:
    assert EXACT_DECK_FINGERPRINT == "b8f251a476e7"
    assert deck_fingerprint(tuple(reversed(EXACT_DECK))) == "b8f251a476e7"


def test_deck_fingerprint_changes_when_one_card_changes() -> None:
    changed = list(EXACT_DECK)
    changed[-1] = 1161
    assert deck_fingerprint(changed) != EXACT_DECK_FINGERPRINT

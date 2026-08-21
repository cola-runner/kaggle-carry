from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterable


COMPETITION = "pokemon-tcg-ai-battle"

EXACT_DECK_COUNTS: dict[int, int] = {
    7: 10,
    112: 4,
    646: 4,
    647: 3,
    648: 3,
    104: 2,
    860: 2,
    1086: 4,
    1152: 4,
    1219: 4,
    1227: 4,
    1259: 4,
    1079: 3,
    1097: 3,
    1182: 2,
    1080: 1,
    1122: 1,
    1137: 1,
    1231: 1,
}


def deck_fingerprint(deck: Iterable[int]) -> str:
    counts = Counter(int(card_id) for card_id in deck)
    text = ",".join(f"{card_id}:{counts[card_id]}" for card_id in sorted(counts))
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


EXACT_DECK = tuple(
    card_id
    for card_id, count in sorted(EXACT_DECK_COUNTS.items())
    for _ in range(count)
)
EXACT_DECK_FINGERPRINT = deck_fingerprint(EXACT_DECK)

SOURCE_WINDOW_HOURS = 72
VALIDATION_WINDOW_HOURS = 12
HOLDOUT_WINDOW_HOURS = 12
MIN_TEACHER_TEAMS = 3
MIN_EPISODES = 500
MIN_MAIN_DECISIONS = 5_000
BRANCH_INTEGRITY_ROOTS = 50
MIN_VALUE_AUC = 0.65
MAX_VALUE_ECE = 0.08
ECE_BINS = 10
MIN_OVERRIDE_ADVANTAGE = 0.10
SAFETY_GAMES = 1_000
MAX_P99_LATENCY_MS = 500.0
MAX_SNAPSHOT_AGE_HOURS = 24
MIN_ONLINE_EPISODES = 200

FORBIDDEN_FEATURE_KEYS = frozenset(
    {
        "visualize",
        "opponent_hand",
        "opponent_deck",
        "opponent_prize_cards",
        "deck_order",
        "hidden",
        "search_begin_input",
    }
)

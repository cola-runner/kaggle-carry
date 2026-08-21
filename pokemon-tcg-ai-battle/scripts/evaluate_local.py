from __future__ import annotations

import argparse
import csv
from pathlib import Path

from run_local_match import run_match


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-a", type=Path, required=True)
    parser.add_argument("--agent-b", type=Path, required=True)
    parser.add_argument("--games-per-seat", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    rows: list[dict[str, str | int]] = []
    wins = {"agent_a": 0, "agent_b": 0, "draw": 0}

    for seat in (0, 1):
        for game_id in range(args.games_per_seat):
            if seat == 0:
                result, decisions = run_match(args.agent_a.resolve(), args.agent_b.resolve(), args.max_steps)
                winner = "agent_a" if result == 0 else "agent_b" if result == 1 else "draw"
            else:
                result, decisions = run_match(args.agent_b.resolve(), args.agent_a.resolve(), args.max_steps)
                winner = "agent_b" if result == 0 else "agent_a" if result == 1 else "draw"

            wins[winner] += 1
            row = {
                "seat": seat,
                "game_id": game_id,
                "result": result,
                "winner": winner,
                "decisions": decisions,
            }
            rows.append(row)
            print(row)

    total = len(rows)
    print(
        "summary "
        f"agent_a_wins={wins['agent_a']} "
        f"agent_b_wins={wins['agent_b']} "
        f"draws={wins['draw']} "
        f"agent_a_winrate={wins['agent_a'] / total:.3f}"
    )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["seat", "game_id", "result", "winner", "decisions"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from league_selfplay.fidelity_runner import run_fidelity, write_report_atomic


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare V1 and V2 driver fidelity without PPO."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--standard", action="store_true")
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports/mac_driver_fidelity_latest.json",
    )
    args = parser.parse_args()

    if args.dry_run:
        games, train_games, epochs = 24, 12, 1
    else:
        games, train_games, epochs = 96, 72, 6
    report = run_fidelity(
        PROJECT_ROOT,
        games=games,
        train_games=train_games,
        seed=20260804,
        epochs=epochs,
        wall_time_seconds=360,
    )
    write_report_atomic(report, args.report)
    print(
        json.dumps(
            {
                "decision": report.decision.code,
                "passed": report.decision.passed,
                "report": str(args.report.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from league_selfplay.contracts import FrozenLeagueConfig
from league_selfplay.runner import run_league, write_report_atomic
from league_selfplay.schedule import build_dry_run_schedule, build_standard_schedule


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the frozen four-policy Pokemon TCG self-play league."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--standard", action="store_true")
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports/mac_four_policy_league_latest.json",
    )
    args = parser.parse_args()

    config = FrozenLeagueConfig()
    schedule = (
        build_dry_run_schedule(config)
        if args.dry_run
        else build_standard_schedule(config)
    )
    report = run_league(config, schedule, PROJECT_ROOT)
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

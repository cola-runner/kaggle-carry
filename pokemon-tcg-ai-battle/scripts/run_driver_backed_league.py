from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from league_selfplay.residual_runner import run_residual_proof, write_report_atomic
from league_selfplay.schedule import JUDGES


def main() -> int:
    parser = argparse.ArgumentParser(description="Run driver-backed self-play on Mac.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--standard", action="store_true")
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports/mac_driver_backed_latest.json",
    )
    args = parser.parse_args()
    judges = JUDGES[:1] if args.dry_run else JUDGES
    report = run_residual_proof(
        PROJECT_ROOT,
        judge_paths=judges,
        wall_time_seconds=240 if args.dry_run else 360,
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

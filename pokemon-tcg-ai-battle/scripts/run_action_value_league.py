from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from league_selfplay.action_value_runner import run_action_value_proof
from league_selfplay.residual_runner import JUDGES, write_report_atomic


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports/mac_action_value_latest.json",
    )
    parser.add_argument("--wall-time-seconds", type=int, default=360)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    judges = JUDGES[:1] if args.dry_run else JUDGES
    report = run_action_value_proof(
        PROJECT_ROOT,
        judge_paths=judges,
        wall_time_seconds=args.wall_time_seconds,
    )
    write_report_atomic(report, args.report)
    summary = {
        "candidate_delta": (
            report.candidate_comparison.delta
            if report.candidate_comparison is not None
            else None
        ),
        "candidate_overrides": report.candidate_overrides,
        "decision": report.decision.code,
        "failures": list(report.failures),
        "promoted_members": [member.value for member in report.promoted_members],
        "report": str(args.report.resolve()),
        "training_explorations": report.training_explorations,
    }
    print(json.dumps(summary, sort_keys=True))
    raise SystemExit(0 if not report.failures else 1)


if __name__ == "__main__":
    main()

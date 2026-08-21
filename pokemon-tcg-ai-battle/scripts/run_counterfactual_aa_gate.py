from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from league_selfplay.counterfactual_gate import (  # noqa: E402
    AAGateReport,
    run_counterfactual_aa_gate,
)


MAX_REPORT_BYTES = 100 * 1024


def write_report(report: AAGateReport, output: Path) -> int:
    payload = json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"
    encoded = payload.encode("utf-8")
    if len(encoded) >= MAX_REPORT_BYTES:
        raise ValueError(
            f"counterfactual gate report is {len(encoded)} bytes; "
            f"limit is {MAX_REPORT_BYTES - 1}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(encoded)
    return len(encoded)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--out",
        type=Path,
        default=(
            PROJECT_ROOT / "reports/mac_counterfactual_aa_engine22_latest.json"
        ),
    )
    parser.add_argument("--samples-per-member", type=int, default=3)
    parser.add_argument("--wall-time-seconds", type=int, default=180)
    args = parser.parse_args()

    report = run_counterfactual_aa_gate(
        args.project_root,
        samples_per_member=args.samples_per_member,
        wall_time_seconds=args.wall_time_seconds,
    )
    report_bytes = write_report(report, args.out)
    summary = report.summary()
    summary["report"] = str(args.out.resolve())
    summary["report_bytes"] = report_bytes
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

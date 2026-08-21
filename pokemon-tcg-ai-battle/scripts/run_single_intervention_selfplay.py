from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from league_selfplay.single_intervention_runner import (  # noqa: E402
    run_single_intervention_proof,
    write_report_atomic,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "reports/mac_single_intervention_selfplay_latest.json",
    )
    parser.add_argument("--wall-time-seconds", type=int, default=600)
    parser.add_argument("--seed", type=int, default=20260804)
    args = parser.parse_args()

    report = run_single_intervention_proof(
        args.project_root,
        wall_time_seconds=args.wall_time_seconds,
        seed=args.seed,
    )
    write_report_atomic(report, args.out)
    print(json.dumps(report.summary(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

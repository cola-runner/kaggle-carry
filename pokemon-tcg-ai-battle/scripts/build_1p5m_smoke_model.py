from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


INPUT_WIDTH = 512
HIDDEN_WIDTH = 1_024
SECOND_WIDTH = 928
EXPECTED_PARAMETERS = 1_477_441


def build_model(output: Path, seed: int = 20260804) -> None:
    rng = np.random.default_rng(seed)
    arrays = {
        "w1": rng.normal(0.0, 0.02, (INPUT_WIDTH, HIDDEN_WIDTH)).astype(np.float32),
        "b1": np.zeros(HIDDEN_WIDTH, dtype=np.float32),
        "w2": rng.normal(0.0, 0.02, (HIDDEN_WIDTH, SECOND_WIDTH)).astype(np.float32),
        "b2": np.zeros(SECOND_WIDTH, dtype=np.float32),
        "wout": rng.normal(0.0, 0.02, SECOND_WIDTH).astype(np.float32),
        "bout": np.zeros(1, dtype=np.float32),
    }
    parameter_count = sum(array.size for array in arrays.values())
    if parameter_count != EXPECTED_PARAMETERS:
        raise RuntimeError(f"unexpected parameter count: {parameter_count}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".npz.tmp")
    with temporary.open("wb") as file:
        np.savez(file, **arrays)
    temporary.replace(output)
    print(f"created {output} parameters={parameter_count} bytes={output.stat().st_size}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260804)
    args = parser.parse_args()
    build_model(args.output.resolve(), args.seed)


if __name__ == "__main__":
    main()


from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable


Agent = Callable[[dict[str, Any]], list[int]]


@contextmanager
def pushd(path: Path):
    old = Path.cwd()
    path_s = str(path)
    os.chdir(path)
    sys.path.insert(0, path_s)
    try:
        yield
    finally:
        try:
            sys.path.remove(path_s)
        except ValueError:
            pass
        os.chdir(old)


def clear_local_helper_modules(agent_dir: Path) -> None:
    for py_file in agent_dir.glob("*.py"):
        if py_file.stem != "main":
            sys.modules.pop(py_file.stem, None)


def import_official_cg(root: Path):
    sample_dir = root / "data/raw/pokemon-tcg-ai-battle/sample_submission/sample_submission"
    if not (sample_dir / "cg").exists():
        fallback_dir = root / "agents/baselines/v11_hammer_metal_from_submission"
        if (fallback_dir / "cg").exists():
            sample_dir = fallback_dir
        else:
            env_dir = os.environ.get("PTCG_CG_DIR")
            if env_dir and (Path(env_dir) / "cg").exists():
                sample_dir = Path(env_dir)
    if not (sample_dir / "cg").exists():
        raise SystemExit(
            "Official cg package not found. Run: "
            "./scripts/download_official_data.sh"
        )
    sys.path.insert(0, str(sample_dir))
    import cg.game as game  # type: ignore

    return game


def load_agent(agent_dir: Path, module_name: str) -> Agent:
    main_py = agent_dir / "main.py"
    if not main_py.exists():
        raise SystemExit(f"missing {main_py}")

    spec = importlib.util.spec_from_file_location(module_name, main_py)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import {main_py}")
    module = importlib.util.module_from_spec(spec)
    clear_local_helper_modules(agent_dir)

    with pushd(agent_dir):
        spec.loader.exec_module(module)

    if not hasattr(module, "agent"):
        raise SystemExit(f"{main_py} does not define agent(obs_dict)")
    return module.agent


def validate_action(action: Any, select: dict[str, Any]) -> list[int]:
    if not isinstance(action, list) or not all(isinstance(i, int) for i in action):
        raise ValueError(f"action must be list[int], got {action!r}")
    option_count = len(select.get("option", []) or [])
    min_count = int(select.get("minCount", 0) or 0)
    max_count = int(select.get("maxCount", 0) or 0)
    if not min_count <= len(action) <= max_count:
        raise ValueError(f"action length {len(action)} not in [{min_count}, {max_count}]")
    if len(set(action)) != len(action):
        raise ValueError(f"action contains duplicates: {action!r}")
    bad = [i for i in action if i < 0 or i >= option_count]
    if bad:
        raise ValueError(f"action out of range for {option_count} options: {bad!r}")
    return action


def first_result_deck(agent: Agent, agent_dir: Path) -> list[int]:
    with pushd(agent_dir):
        deck = agent({"select": None, "logs": [], "current": None, "search_begin_input": None})
    if not isinstance(deck, list) or len(deck) != 60 or not all(isinstance(i, int) for i in deck):
        raise ValueError(f"initial agent result is not a 60-card deck: {deck!r}")
    return deck


def run_match(agent0_dir: Path, agent1_dir: Path, max_steps: int) -> tuple[int, int]:
    root = Path(__file__).resolve().parents[1]
    game = import_official_cg(root)

    agent0 = load_agent(agent0_dir, "agent0_main")
    agent1 = load_agent(agent1_dir, "agent1_main")
    agents = [agent0, agent1]
    agent_dirs = [agent0_dir, agent1_dir]

    deck0 = first_result_deck(agent0, agent0_dir)
    deck1 = first_result_deck(agent1, agent1_dir)

    obs, start_data = game.battle_start(deck0, deck1)
    if obs is None:
        raise RuntimeError(
            f"battle_start failed: errorPlayer={start_data.errorPlayer}, "
            f"errorType={start_data.errorType}"
        )

    try:
        for step in range(max_steps):
            current = obs["current"]
            result = current.get("result", -1)
            if result >= 0:
                return result, step

            player = int(current["yourIndex"])
            with pushd(agent_dirs[player]):
                action = agents[player](obs)
            action = validate_action(action, obs["select"])
            obs = game.battle_select(action)

        raise RuntimeError(f"match did not finish within {max_steps} decisions")
    finally:
        game.battle_finish()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent0", type=Path, required=True)
    parser.add_argument("--agent1", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=2000)
    args = parser.parse_args()

    result, steps = run_match(args.agent0.resolve(), args.agent1.resolve(), args.max_steps)
    if result == 0:
        outcome = "agent0_win"
    elif result == 1:
        outcome = "agent1_win"
    else:
        outcome = "draw"
    print(f"result={result} outcome={outcome} decisions={steps}")


if __name__ == "__main__":
    main()

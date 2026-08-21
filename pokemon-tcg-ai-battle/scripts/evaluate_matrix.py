from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from run_local_match import run_match


def parse_agent(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.name, path
    name, path = value.split("=", 1)
    return name, Path(path)


def run_one(task: tuple[str, str, str, str, int, int, int]) -> dict[str, int | str]:
    agent_name, agent_dir_s, opponent_name, opponent_dir_s, seat, game_id, max_steps = task
    agent_dir = Path(agent_dir_s)
    opponent_dir = Path(opponent_dir_s)

    try:
        if seat == 0:
            result, decisions = run_match(agent_dir, opponent_dir, max_steps)
            win = int(result == 0)
        else:
            result, decisions = run_match(opponent_dir, agent_dir, max_steps)
            win = int(result == 1)
    except Exception as exc:
        raise RuntimeError(
            f"match failed agent={agent_name} opponent={opponent_name} "
            f"seat={seat} game_id={game_id}"
        ) from exc

    return {
        "agent": agent_name,
        "opponent": opponent_name,
        "seat": seat,
        "game_id": game_id,
        "win": win,
        "result": result,
        "decisions": decisions,
    }


def run_one_subprocess(task: tuple[str, str, str, str, int, int, int]) -> dict[str, int | str]:
    agent_name, agent_dir_s, opponent_name, opponent_dir_s, seat, game_id, max_steps = task
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    if seat == 0:
        agent0_dir, agent1_dir = agent_dir_s, opponent_dir_s
        win_result = 0
    else:
        agent0_dir, agent1_dir = opponent_dir_s, agent_dir_s
        win_result = 1

    try:
        proc = subprocess.run(
            [
                sys.executable,
                str(script_dir / "run_local_match.py"),
                "--agent0",
                agent0_dir,
                "--agent1",
                agent1_dir,
                "--max-steps",
                str(max_steps),
            ],
            cwd=project_root,
            text=True,
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"match subprocess failed agent={agent_name} opponent={opponent_name} "
            f"seat={seat} game_id={game_id}\nstdout={exc.stdout}\nstderr={exc.stderr}"
        ) from exc
    match = re.search(r"result=(-?\d+).*decisions=(\d+)", proc.stdout)
    if match is None:
        raise RuntimeError(f"could not parse run_local_match output: {proc.stdout!r}")
    result = int(match.group(1))
    decisions = int(match.group(2))
    return {
        "agent": agent_name,
        "opponent": opponent_name,
        "seat": seat,
        "game_id": game_id,
        "win": int(result == win_result),
        "result": result,
        "decisions": decisions,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", action="append", required=True, help="name=path")
    parser.add_argument("--opponent", action="append", required=True, help="name=path")
    parser.add_argument("--games-per-seat", type=int, default=25)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--isolate", action="store_true", help="restart worker processes between matches")
    parser.add_argument("--subprocess", action="store_true", help="run each match through run_local_match.py")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    agents = [parse_agent(value) for value in args.agent]
    opponents = [parse_agent(value) for value in args.opponent]
    tasks: list[tuple[str, str, str, str, int, int, int]] = []
    for agent_name, agent_dir in agents:
        for opponent_name, opponent_dir in opponents:
            for seat in (0, 1):
                for game_id in range(args.games_per_seat):
                    tasks.append(
                        (
                            agent_name,
                            str(agent_dir.resolve()),
                            opponent_name,
                            str(opponent_dir.resolve()),
                            seat,
                            game_id,
                            args.max_steps,
                        )
                    )

    rows: list[dict[str, int | str]] = []
    runner = run_one_subprocess if args.subprocess else run_one
    if args.jobs == 1:
        for index, task in enumerate(tasks, 1):
            rows.append(runner(task))
            if index % 100 == 0:
                print(f"done {index}/{len(tasks)}", flush=True)
    else:
        chunksize = 1 if args.isolate else 4
        with mp.Pool(processes=args.jobs, maxtasksperchild=1 if args.isolate else None) as pool:
            for index, row in enumerate(pool.imap_unordered(runner, tasks, chunksize=chunksize), 1):
                rows.append(row)
                if index % 100 == 0:
                    print(f"done {index}/{len(tasks)}", flush=True)

    rows.sort(key=lambda row: (str(row["agent"]), str(row["opponent"]), int(row["seat"]), int(row["game_id"])))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["agent", "opponent", "seat", "game_id", "win", "result", "decisions"])
        writer.writeheader()
        writer.writerows(rows)

    by_agent: defaultdict[str, list[int]] = defaultdict(lambda: [0, 0])
    by_opponent: defaultdict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    for row in rows:
        agent = str(row["agent"])
        opponent = str(row["opponent"])
        win = int(row["win"])
        by_agent[agent][0] += win
        by_agent[agent][1] += 1
        by_opponent[(agent, opponent)][0] += win
        by_opponent[(agent, opponent)][1] += 1

    print(f"wrote {args.out}")
    for agent, (wins, total) in sorted(by_agent.items(), key=lambda item: (-item[1][0] / item[1][1], item[0])):
        print(f"total {agent} {wins}/{total} {wins / total:.3f}")
    print("--- by opponent")
    for key, (wins, total) in sorted(by_opponent.items()):
        print(f"opp {key[0]} {key[1]} {wins}/{total} {wins / total:.3f}")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()

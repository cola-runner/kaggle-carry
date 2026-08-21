from __future__ import annotations

import argparse
import tarfile
from pathlib import Path, PurePosixPath


def read_deck(deck_path: Path) -> list[int]:
    rows: list[int] = []
    for line_no, raw in enumerate(deck_path.read_text().splitlines(), start=1):
        text = raw.strip()
        if not text:
            continue
        try:
            rows.append(int(text))
        except ValueError as exc:
            raise SystemExit(f"{deck_path}:{line_no}: deck row is not an integer: {text!r}") from exc
    if len(rows) != 60:
        raise SystemExit(f"{deck_path}: expected 60 card IDs, got {len(rows)}")
    return rows


def clean_tar_member(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
    parts = PurePosixPath(info.name).parts
    if "__pycache__" in parts or info.name.endswith(".pyc") or info.name.endswith(".DS_Store"):
        return None
    return info


def add_path(tar: tarfile.TarFile, path: Path, arcname: str) -> None:
    tar.add(path, arcname=arcname, recursive=path.is_dir(), filter=clean_tar_member)


def official_cg_dir(project_root: Path) -> Path | None:
    path = project_root / "data/raw/pokemon-tcg-ai-battle/sample_submission/sample_submission/cg"
    return path if path.exists() else None


def build_archive(agent_dir: Path, output: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    main_py = agent_dir / "main.py"
    deck_csv = agent_dir / "deck.csv"
    if not main_py.exists():
        raise SystemExit(f"missing {main_py}")
    if not deck_csv.exists():
        raise SystemExit(f"missing {deck_csv}; copy a legal deck into this path first")
    read_deck(deck_csv)

    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz") as tar:
        add_path(tar, main_py, "main.py")
        add_path(tar, deck_csv, "deck.csv")
        for py_file in sorted(agent_dir.glob("*.py")):
            if py_file.name != "main.py":
                add_path(tar, py_file, py_file.name)

        for package_dir_name in (
            "runtime",
            "models",
            "si_alakazam",
            "si_grimmsnarl",
        ):
            package_dir = agent_dir / package_dir_name
            if package_dir.exists():
                add_path(tar, package_dir, package_dir_name)

        needs_cg = "import cg" in main_py.read_text() or "from cg" in main_py.read_text()
        if needs_cg:
            cg_dir = agent_dir / "cg"
            if not cg_dir.exists():
                cg_dir = official_cg_dir(project_root)
            if cg_dir is None or not cg_dir.exists():
                raise SystemExit("main.py imports cg, but no cg/ package was found")
            add_path(tar, cg_dir, "cg")

        for extra_name in (
            "model.pkl",
            "model.joblib",
            "model.npz",
            "choice_ranker.json",
            "choice_tree.json",
            "choice_tree_crustle.json",
            "value_model.json",
            "identity.json",
        ):
            extra = agent_dir / extra_name
            if extra.exists():
                add_path(tar, extra, extra_name)

    print(f"created {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("submissions/submission.tar.gz"))
    args = parser.parse_args()
    build_archive(args.agent_dir, args.output)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run the KITE main method and optional baselines from config.json."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def load_experiment_names(config_path: Path, selected: list[str] | None) -> list[str]:
    config = json.loads(config_path.read_text())
    experiments = config.get("experiments", {})
    if selected:
        missing = [name for name in selected if name not in experiments]
        if missing:
            choices = ", ".join(sorted(experiments))
            raise ValueError(f"Unknown experiment(s): {', '.join(missing)}. Choices: {choices}")
        return selected
    main_method = config.get("main_method") or config.get("default_experiment")
    baselines = config.get("baseline_methods", [])
    ordered = []
    for name in [main_method, *baselines]:
        if name in experiments and name not in ordered:
            ordered.append(name)
    for name in experiments:
        if name not in ordered:
            ordered.append(name)
    return ordered


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the KITE main method and optional baselines sequentially.")
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/kite"))
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=None,
        help="Optional subset of method names. Defaults to the main KITE method followed by baselines.",
    )
    args = parser.parse_args()

    script = Path(__file__).resolve().with_name("run_kite.py")
    experiment_names = load_experiment_names(args.config, args.experiments)

    for experiment_name in experiment_names:
        cmd = [
            sys.executable,
            str(script),
            "--config",
            str(args.config),
            "--experiment",
            experiment_name,
            "--model-root",
            str(args.model_root),
            "--parquet",
            str(args.parquet),
            "--output-root",
            str(args.output_root),
            "--start",
            str(args.start),
        ]
        if args.limit is not None:
            cmd.extend(["--limit", str(args.limit)])
        if args.max_new_tokens is not None:
            cmd.extend(["--max-new-tokens", str(args.max_new_tokens)])

        print("=" * 72, flush=True)
        print(f"Running KITE method: {experiment_name}", flush=True)
        print("=" * 72, flush=True)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

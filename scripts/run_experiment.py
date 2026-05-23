#!/usr/bin/env python
"""Run one or all ML experiments from YAML configs.

Examples:
  python scripts/run_experiment.py --config configs/emission_photon_energy/baseline_rfecv.yaml
  python scripts/run_experiment.py --all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from ml_quasit.experiment_config import load_config  # noqa: E402
from ml_quasit.run_experiment import run_experiment  # noqa: E402


def list_configs() -> list[Path]:
    return sorted((REPO / "configs").rglob("*.yaml"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ML Quasit 2D perovskite experiments")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to a YAML config under configs/",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run every config in configs/",
    )
    args = parser.parse_args()

    if args.all:
        paths = list_configs()
    elif args.config:
        paths = [Path(args.config)]
        if not paths[0].is_absolute():
            paths[0] = REPO / paths[0]
    else:
        parser.print_help()
        sys.exit(1)

    for path in paths:
        config = load_config(path)
        run_experiment(config)


if __name__ == "__main__":
    main()

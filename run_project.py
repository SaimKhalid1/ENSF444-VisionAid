"""Command-line entrypoint for the VisionAid ENSF 444 project."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from visionaid import FaceShapeExperiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the VisionAid face-shape classification and eyewear recommendation project."
    )
    parser.add_argument(
        "--force-features",
        action="store_true",
        help="Rebuild landmark features even if a cached CSV already exists.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    experiment = FaceShapeExperiment()
    metrics_df, summary = experiment.run_all(force_rebuild_features=args.force_features)

    print("Project run complete.")
    print(metrics_df.to_string(index=False))
    print()
    print("Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


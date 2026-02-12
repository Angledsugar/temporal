#!/usr/bin/env python3
"""Convert Inter-X human pose dataset to TempoRAL-compatible formats.

Usage:
    # NPZ format (for direct pipeline use)
    python scripts/convert_interx.py --format npz \
        --interx-root /media/engineer/DATA/datasets/interx \
        --output-dir dataset/interx_npz \
        --fps 30

    # LeRobot format (for Hub sharing)
    python scripts/convert_interx.py --format lerobot \
        --interx-root /media/engineer/DATA/datasets/interx \
        --output-dir dataset/interx_lerobot \
        --fps 30

    # Stats only (diagnostic)
    python scripts/convert_interx.py --format stats \
        --interx-root /media/engineer/DATA/datasets/interx \
        --max-sequences 50

    # Small test run
    python scripts/convert_interx.py --format npz \
        --interx-root /media/engineer/DATA/datasets/interx \
        --output-dir dataset/interx_test \
        --max-sequences 10
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Convert Inter-X dataset to TempoRAL formats")
    parser.add_argument(
        "--format",
        choices=["npz", "lerobot", "stats"],
        required=True,
        help="Output format: npz (DROID-compatible), lerobot (Hub-compatible), stats (diagnostics only)",
    )
    parser.add_argument(
        "--interx-root",
        type=str,
        required=True,
        help="Path to Inter-X dataset root directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (required for npz/lerobot)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Target FPS (source is 120fps). Default: 30",
    )
    parser.add_argument(
        "--hand-mode",
        choices=["actor", "both", "p1"],
        default="actor",
        help="Which person/hand to extract. 'actor': use interaction_order, 'both': all, 'p1': P1 only",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Convert only this split (train/val/test). Default: all splits",
    )
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=None,
        help="Limit number of sequences per split (for testing)",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="temporal/interx",
        help="LeRobot repo ID (for lerobot format)",
    )
    parser.add_argument(
        "--gripper-threshold",
        type=float,
        default=0.04,
        help="Thumb-index distance threshold for gripper closed (meters)",
    )

    args = parser.parse_args()

    if args.format in ("npz", "lerobot") and not args.output_dir:
        parser.error("--output-dir is required for npz and lerobot formats")

    from temporal.data.interx_converter import InterXConverter

    converter = InterXConverter(
        interx_root=args.interx_root,
        target_fps=args.fps,
        hand_mode=args.hand_mode,
        gripper_threshold=args.gripper_threshold,
    )

    logger.info(f"Inter-X root: {converter.root}")
    logger.info(f"Skeleton dir: {converter.skeleton_dir}")
    logger.info(f"Text dir: {converter.text_dir}")
    logger.info(f"Splits: { {k: len(v) for k, v in converter.splits.items()} }")
    logger.info(f"Target FPS: {args.fps}, Hand mode: {args.hand_mode}")

    if args.format == "stats":
        stats = converter.compute_stats(max_sequences=args.max_sequences or 100)
        print(json.dumps(stats, indent=2))
        return

    if args.format == "npz":
        counts = converter.convert_to_npz(
            output_dir=args.output_dir,
            split=args.split,
            max_sequences=args.max_sequences,
        )
        logger.info(f"NPZ conversion complete: {counts}")

    elif args.format == "lerobot":
        converter.convert_to_lerobot(
            output_dir=args.output_dir,
            repo_id=args.repo_id,
            split=args.split,
            max_sequences=args.max_sequences,
        )
        logger.info("LeRobot conversion complete")


if __name__ == "__main__":
    main()

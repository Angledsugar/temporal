"""CLI entry point for ``temporal-download`` command.

Registered in pyproject.toml as::

    [project.scripts]
    temporal-download = "temporal.utils.download_cli:main"

Usage:
    uv run temporal-download                        # download everything
    uv run temporal-download --list                 # show available assets
    uv run temporal-download --dataset              # datasets only
    uv run temporal-download --checkpoint           # checkpoints only
    uv run temporal-download droid_100 pi05_droid   # specific assets
    uv run temporal-download --dest /data --force   # custom root, re-download
"""

from __future__ import annotations

import argparse
import logging
import sys

from rich.console import Console
from rich.table import Table

from temporal.utils.download import ASSETS, download_all, download_asset, list_assets

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def _print_asset_table() -> None:
    console = Console()
    table = Table(title="Available Assets")
    table.add_column("Name", style="cyan")
    table.add_column("Category", style="green")
    table.add_column("Description")
    table.add_column("Source", style="dim")
    table.add_column("Default Dest", style="yellow")
    for asset in list_assets():
        table.add_row(
            asset["name"],
            asset["category"],
            asset["description"],
            asset["source"],
            asset["default_dest"],
        )
    console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download TempoRAL datasets and checkpoints",
    )
    parser.add_argument(
        "assets",
        nargs="*",
        help=f"Asset names to download. Available: {', '.join(sorted(ASSETS))}",
    )
    parser.add_argument("--list", action="store_true", help="List available assets and exit")
    parser.add_argument("--dataset", action="store_true", help="Download datasets only")
    parser.add_argument("--checkpoint", action="store_true", help="Download checkpoints only")
    parser.add_argument("--dest", type=str, default=None, help="Root destination directory")
    parser.add_argument("--force", action="store_true", help="Re-download even if already present")
    args = parser.parse_args()

    if args.list:
        _print_asset_table()
        sys.exit(0)

    # Specific assets requested
    if args.assets:
        for name in args.assets:
            download_asset(name, dest=args.dest, force=args.force)
        return

    # Category filter
    categories = None
    if args.dataset or args.checkpoint:
        categories = []
        if args.dataset:
            categories.append("dataset")
        if args.checkpoint:
            categories.append("checkpoint")

    results = download_all(dest_root=args.dest, categories=categories, force=args.force)

    if results:
        logger.info("All downloads complete:")
        for name, path in results.items():
            logger.info("  %s -> %s", name, path)
    else:
        logger.info("Nothing to download.")


if __name__ == "__main__":
    main()

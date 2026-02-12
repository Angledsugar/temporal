#!/usr/bin/env python3
"""Download TempoRAL datasets and checkpoints.

Usage:
    uv run python scripts/download_assets.py                        # download everything
    uv run python scripts/download_assets.py --list                 # show available assets
    uv run python scripts/download_assets.py --dataset              # datasets only
    uv run python scripts/download_assets.py --checkpoint           # checkpoints only
    uv run python scripts/download_assets.py droid_100 pi05_droid   # specific assets
    uv run python scripts/download_assets.py --dest /data --force   # custom root, re-download

Or via the registered entry point:
    uv run temporal-download [same arguments]
"""

from temporal.utils.download_cli import main

if __name__ == "__main__":
    main()

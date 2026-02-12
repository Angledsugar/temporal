"""Download datasets and checkpoints for TempoRAL."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Registry: each entry defines a downloadable asset
# ---------------------------------------------------------------------------

ASSETS: dict[str, dict] = {
    # Datasets
    "droid_100": {
        "description": "DROID droid_100 robot manipulation dataset",
        "source": "gs://gresearch/robotics/droid_100",
        "default_dest": "dataset/droid_100",
        "category": "dataset",
        "method": "gsutil",
    },
    # Checkpoints
    "pi05_droid": {
        "description": "pi0.5-DROID pretrained checkpoint (JAX/Orbax)",
        "source": "gs://openpi-assets/checkpoints/pi05_droid",
        "default_dest": "checkpoint/pi05_droid",
        "category": "checkpoint",
        "method": "gsutil",
    },
}


def _require_gsutil() -> str:
    """Return path to gsutil, raising if not found."""
    path = shutil.which("gsutil")
    if path is None:
        raise RuntimeError(
            "gsutil is not installed or not on PATH.\n"
            "Install it via: https://cloud.google.com/storage/docs/gsutil_install"
        )
    return path


def _download_gsutil(source: str, dest: Path) -> None:
    """Download a GCS directory using gsutil."""
    gsutil = _require_gsutil()
    dest.mkdir(parents=True, exist_ok=True)
    cmd = [gsutil, "-m", "cp", "-r", f"{source}/*", str(dest) + "/"]
    logger.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _dest_exists(dest: Path) -> bool:
    """Check whether destination already contains files."""
    return dest.is_dir() and any(dest.iterdir())


def download_asset(
    name: str,
    dest: str | Path | None = None,
    *,
    force: bool = False,
) -> Path:
    """Download a single asset by name.

    Args:
        name: Key in the ASSETS registry (e.g. ``"droid_100"``).
        dest: Override destination directory. Defaults to ``ASSETS[name]["default_dest"]``.
        force: Re-download even if destination already exists.

    Returns:
        Path to the downloaded directory.
    """
    if name not in ASSETS:
        available = ", ".join(sorted(ASSETS))
        raise ValueError(f"Unknown asset '{name}'. Available: {available}")

    info = ASSETS[name]
    dest = Path(dest) if dest else Path(info["default_dest"])

    if _dest_exists(dest) and not force:
        logger.info("[skip] %s already exists at %s", name, dest)
        return dest

    logger.info("[download] %s -> %s", info["description"], dest)

    method = info["method"]
    if method == "gsutil":
        _download_gsutil(info["source"], dest)
    else:
        raise ValueError(f"Unknown download method: {method}")

    size = _dir_size_human(dest)
    logger.info("[done] %s  (%s)", name, size)
    return dest


def download_all(
    dest_root: str | Path | None = None,
    *,
    categories: list[str] | None = None,
    force: bool = False,
) -> dict[str, Path]:
    """Download all (or filtered) assets.

    Args:
        dest_root: Base directory prepended to each asset's default dest.
        categories: If set, only download assets whose category matches
                    (e.g. ``["dataset"]``, ``["checkpoint"]``, or both).
        force: Re-download even if destinations already exist.

    Returns:
        Mapping of asset name to its local path.
    """
    results: dict[str, Path] = {}
    for name, info in ASSETS.items():
        if categories and info["category"] not in categories:
            continue
        dest = Path(dest_root, info["default_dest"]) if dest_root else None
        results[name] = download_asset(name, dest=dest, force=force)
    return results


def list_assets() -> list[dict]:
    """Return a list of all registered assets with metadata."""
    return [{"name": k, **v} for k, v in ASSETS.items()]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dir_size_human(path: Path) -> str:
    """Return human-readable size of a directory."""
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if total < 1024:
            return f"{total:.1f} {unit}"
        total /= 1024
    return f"{total:.1f} PB"

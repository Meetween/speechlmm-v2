"""Register ShareGPT datasets into a LLaMA-Factory ``dataset_info.json``.

Registration is decoupled from creation: ``parquet_to_sharegpt`` / ``CorpusPreparer``
write the per-corpus/split/task ShareGPT JSONs, and this module turns any set of those
JSONs into a ``dataset_info.json``. It backs both the per-corpus registration done inside
``CorpusPreparer.prepare`` and a standalone CLI that scans a directory tree to register
many corpora into a single registry (e.g. to assemble a training mix).

``file_name`` entries are stored relative to the directory the registry is written to, so
LLaMA-Factory (which resolves ``file_name`` against the registry location) finds each JSON.

Usage (combine every corpus under a root into one dataset_info.json):
    PYTHONPATH=src python3 -m speechlmm.data.registry_dataset \\
        --from-dir $DATASETS_DIR/speechlmm-v2
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_SHAREGPT_TAGS = {
    "role_tag": "role",
    "content_tag": "content",
    "user_tag": "user",
    "assistant_tag": "assistant",
}

_SHAREGPT_DIRS = ("sharegpt", "json")


def columns_for_entries(entries: list[dict]) -> dict[str, str]:
    """Map only columns that exist in the ShareGPT entries (avoids KeyError in LF loader)."""
    cols: dict[str, str] = {"messages": "messages"}
    for key in ("audios", "images", "videos", "lipread", "codec_tokens", "speaker", "sample_id"):
        if any(key in e for e in entries):
            cols[key] = key
    return cols


def generate_dataset_info(datasets: dict[str, str], root: Path) -> dict:
    """Build dataset_info.json fragments, inferring columns from each ShareGPT file.

    Args:
        datasets: ``{dataset_key: json_path}`` where ``json_path`` is relative to ``root``
            and is recorded verbatim as ``file_name``.
        root: directory the json paths (and the resulting registry) are relative to.
    """
    root = Path(root)
    info: dict[str, dict] = {}
    for name, rel_path in datasets.items():
        json_path = root / rel_path
        try:
            entries = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot register dataset {name!r} from {json_path}") from exc
        if not isinstance(entries, list):
            raise ValueError(f"Dataset {name!r} must contain a JSON list: {json_path}")
        if not entries:
            logger.warning("Skipping empty dataset %s (%s)", name, json_path)
            continue
        info[name] = {
            "file_name": rel_path,
            "formatting": "sharegpt",
            "columns": columns_for_entries(entries),
            "tags": dict(_SHAREGPT_TAGS),
        }
    return info


def collect_datasets(root: Path, use_filtered: bool = False, use_small: bool = False) -> dict[str, str]:
    """Find every ShareGPT JSON under ``root`` -> ``{dataset_key: path relative to root}``."""
    root = Path(root)
    json_files: list[Path] = []

    prefix = "small_" if use_small else ""
    prefix = f"{prefix}filtered_" if use_filtered else prefix
    for subdir_name in _SHAREGPT_DIRS:
        for subdir in (p for p in root.rglob(f"{prefix}{subdir_name}*") if p.is_dir()):
            json_files.extend(p for p in subdir.glob("*.json") if p.name != "dataset_info.json")

    datasets: dict[str, str] = {}
    for path in sorted(json_files):
        relative_path = path.relative_to(root).as_posix()
        if path.stem in datasets and datasets[path.stem] != relative_path:
            raise ValueError(
                f"Duplicate dataset key {path.stem!r}: {datasets[path.stem]} and {relative_path}"
            )
        datasets[path.stem] = relative_path
    return datasets


def write_registry_dataset(
    datasets: dict[str, str],
    root: Path,
    *,
    replace: bool = False,
) -> None:
    """Write/merge ``datasets`` into ``<root>/dataset_info.json`` (paths relative to ``root``)."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    info = generate_dataset_info(datasets, root)
    info_path = root / "dataset_info.json"
    if info_path.exists() and not replace:
        existing = json.loads(info_path.read_text(encoding="utf-8"))
        existing.update(info)
        info = existing
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Updated %s with %d entries", info_path, len(info))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Register ShareGPT datasets into a single dataset_info.json."
    )
    parser.add_argument(
        "--from-dir",
        type=Path,
        required=True,
        help="Root scanned recursively for 'sharegpt'/'json' dirs; dataset_info.json is "
        "written here and file_name paths are stored relative to it.",
    )
    parser.add_argument("--filter", action="store_true", help="Take into account filtered data")
    parser.add_argument("--small", action="store_true", help="Take into account small data")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    datasets = collect_datasets(args.from_dir, use_filtered=args.filter, use_small=args.small)
    if not datasets:
        raise SystemExit(f"No ShareGPT datasets found under {args.from_dir}")
    logger.info("Found %d dataset(s) under %s", len(datasets), args.from_dir)
    write_registry_dataset(datasets, args.from_dir, replace=True)


if __name__ == "__main__":
    main()

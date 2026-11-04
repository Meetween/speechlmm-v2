"""Download FLEURS audio files and text files for specified splits.

Uses the official HuggingFace FLEURS mirror (same source as Kaldi, Lhotse, ESPnet).
Default split is "train".

Usage:
    PYTHONPATH=src python3 -m speechlmm.data.processing.fleurs.download \\
        --tsv-dir $SCRATCH/data/fleurs/tsv \\
        --signals-dir $SCRATCH/data/fleurs/signals
"""

from __future__ import annotations

import argparse
import logging
import urllib.error
import urllib.request
from pathlib import Path

from speechlmm.data.utils import datasets_dir

logger = logging.getLogger(__name__)

FLEURS_MIRROR = "https://huggingface.co/datasets/google/fleurs/resolve/main/data"
LICENSE_URL = "https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/cc-by-4.0.md"
SOURCE = "fleurs"

LANGUAGES = {
    "cs": "cs_cz",
    "de": "de_de",
    "en": "en_us",
    "es": "es_419",
    "fr": "fr_fr",
    "hu": "hu_hu",
    "it": "it_it",
    "nl": "nl_nl",
    "pt": "pt_br",
    "sv": "sv_se",
}

SPLITS = {
    "train": {"audios": {"filename": "train.tar.gz", "subdir": "audio"}, "texts": {"filename": "train.tsv"}},
    "dev": {"audios": {"filename": "dev.tar.gz", "subdir": "audio"}, "texts": {"filename": "dev.tsv"}},
    "test": {"audios": {"filename": "test.tar.gz", "subdir": "audio"}, "texts": {"filename": "test.tsv"}},
}


def download_tsv(tsv_dir: Path, splits: list[str], languages: list[str], base_url: str) -> None:
    for split in splits:
        for language in languages:
            _, err = _download_tsv(tsv_dir, split, language, base_url=base_url)
            if err != "ok":
                logger.error("Error downloading TSV for %s/%s: %s", split, language, err)
                continue


def _download_tsv(tsv_dir: Path, split: str, language: str, base_url: str) -> tuple[Path, str]:
    spec = SPLITS[split]["texts"]
    filename = spec["filename"]
    url = f"{base_url}/{LANGUAGES[language]}/{filename}"
    dest = tsv_dir / language / filename
    dest.parent.mkdir(parents=True, exist_ok=True)

    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "speechlmm-fleurs-downloader/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            tmp.write_bytes(resp.read())
        tmp.replace(dest)
    except urllib.error.HTTPError as exc:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        return dest, f"error HTTP {exc.code}"
    except Exception as exc:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        return dest, f"error {exc}"

    logger.info(f"Downloaded {split} TSV for language {language} ({LANGUAGES[language]}) to {dest}")

    return dest, "ok"


def download_audios(signals_dir: Path, splits: list, languages: list) -> None:
    for split in splits:
        for language in languages:
            download_audio_archive(signals_dir, split, language, base_url=FLEURS_MIRROR, force=False)
            unpack_audio_archive(signals_dir, split, language)


def download_audio_archive(output_dir: Path, split: str, language: str, base_url: str, force: bool = False) -> tuple[Path, str]:
    spec = SPLITS[split]["audios"]
    filename = spec["filename"]
    subdir = spec["subdir"]
    url = f"{base_url}/{LANGUAGES[language]}/{subdir}/{filename}"
    dest = output_dir / subdir / language / filename
    dest.parent.mkdir(parents=True, exist_ok=True)

    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "speechlmm-fleurs-downloader/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            tmp.write_bytes(resp.read())
        tmp.replace(dest)
        logger.info("Wrote %s", dest)
        return dest, "ok"
    except urllib.error.HTTPError as exc:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        return dest, f"error HTTP {exc.code}"
    except Exception as exc:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        return dest, f"error {exc}"


def download_license(output_dir: Path) -> None:
    dest = output_dir / "LICENCE-cc-by-4.0.md"
    if dest.exists():
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(LICENSE_URL, headers={"User-Agent": "speechlmm-fleurs-downloader/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        dest.write_bytes(resp.read())
    logger.info("Wrote %s", dest)


def unpack_audio_archive(output_dir: Path, split: str, language: str, force: bool = False) -> None:
    spec = SPLITS[split]["audios"]
    filename = spec["filename"]
    subdir = spec["subdir"]
    archive_path = output_dir / subdir / language / filename
    if not archive_path.exists():
        logger.warning("Archive %s does not exist, cannot unpack", archive_path)
        return
    import tarfile

    output_dir = output_dir / subdir / language
    with tarfile.open(archive_path, "r:gz") as tar:
        try:
            tar.extractall(path=output_dir)
        except (tarfile.TarError, OSError, EOFError) as exc:
            logger.error("Error during extraction: %s", exc)
        except Exception as exc:
            logger.error("Generic Exception during extraction: %s", exc)
    logger.info("Unpacked %s to %s", archive_path, output_dir)

    archive_path.unlink(missing_ok=True)


def download_signals(
    tsv_dir: Path,
    parquet_dir: Path,
    output_dir: Path,
    *,
    split: str = "train",
    language: str = "en",
    base_url: str = FLEURS_MIRROR,
    workers: int = 4,
    force: bool = False,
) -> None:
    """Download the audio files for the specified split and language"""
    tsv_dir = Path(tsv_dir)
    parquet_dir = Path(parquet_dir)
    output_dir = Path(output_dir)
    tsv_dir.mkdir(parents=True, exist_ok=True)
    parquet_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    download_tsv(tsv_dir, [split], [language], base_url)

    download_license(output_dir)

    download_audio_archive(output_dir, split, language, base_url, force=force)

    unpack_audio_archive(output_dir, split, language, force=force)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download FLEURS audio signals for Parquet splits.")
    parser.add_argument(
        "--parquet-dir",
        type=Path,
        default=None,
        help="Directory with train/dev/test.parquet",
    )
    parser.add_argument(
        "--signals-dir",
        type=Path,
        default=None,
        help="Root for signals; writes {output_dir}/audio/*.wav",
    )
    parser.add_argument(
        "--tsv-dir",
        type=Path,
        default=None,
        help="Root for tsv; writes {output_dir}/tsv/*.tsv",
    )
    parser.add_argument(
        "--language",
        choices=tuple(LANGUAGES.keys()),
        default="en",
        help="FLEURS language (default: en)",
    )
    parser.add_argument(
        "--split",
        choices=tuple(SPLITS.keys()),
        default="train",
        help="FLEURS split (default: train)",
    )
    parser.add_argument("--base-url", default=FLEURS_MIRROR, help="FLEURS corpus mirror base URL")
    parser.add_argument("--workers", type=int, default=4, help="Parallel download workers")
    parser.add_argument("--force", action="store_true", help="Re-download even if file exists")
    parser.add_argument("--dry-run", action="store_true", help="Print URLs only, do not download")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    default_dataset_root = datasets_dir() / SOURCE
    tsv_dir = args.tsv_dir or default_dataset_root / "tsv"
    parquet_dir = args.parquet_dir or default_dataset_root/ "parquet"
    output_dir = args.signals_dir or default_dataset_root / "signals"
    download_signals(
        tsv_dir,
        parquet_dir,
        output_dir,
        split=args.split,
        language=args.language,
        base_url=args.base_url,
        workers=args.workers,
        force=args.force,
    )


if __name__ == "__main__":
    main()

"""Merge multiple ShareGPT JSON arrays into one file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

_MEDIA_KEYS = ("audios", "videos", "lipread", "images")


def fix_links(entries: list[dict], prefix: str) -> None:
    for entry in entries:
        for key in _MEDIA_KEYS:
            if key in entry:
                entry[key] = [f"{prefix}/{path}" for path in entry[key]]

def main() -> None:
    parser = argparse.ArgumentParser(description="Merge ShareGPT JSON files.")
    parser.add_argument("--common-dir", type=Path)
    parser.add_argument("--output-file", type=Path)
    parser.add_argument("--input-files", nargs="+", type=Path)
    args = parser.parse_args()


    input_files = args.input_files
    common_dir = args.common_dir
    output_file = args.output_file

    merged: list[dict] = []
    for input_file in input_files:
        prefix = input_file.relative_to(common_dir).parts[0]
        data = json.loads(input_file.read_text(encoding="utf-8"))
        fix_links(data, prefix)
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON list in {input_file}")
        merged.extend(data)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f'The file {output_file} aggregates the following {len(input_files)} files:')
    print(f'{"\n".join([str(file) for file in input_files])}')
    print(f"Merged {len(merged)} samples into {output_file}")
    print()

if __name__ == "__main__":
    main()

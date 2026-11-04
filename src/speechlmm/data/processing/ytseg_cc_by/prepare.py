"""YTSeg (CC BY) corpus preparation pipeline.

Commercially-usable subset of YTSeg: the ~1.7% of videos whose original YouTube
license is CC BY, released under CC BY 4.0. Only the ``train`` split exists.
Everything else is inherited from :class:`YTSegCorpusPreparer`.
"""

from __future__ import annotations

import argparse
import logging

from speechlmm.data.processing.ytseg.prepare import YTSegCorpusPreparer, add_common_args


class YTSegCCByCorpusPreparer(YTSegCorpusPreparer):
    name = "ytseg_cc_by"
    dataset_id = "retkowski/ytseg-cc-by"
    license = "CC-BY-4.0"
    source = "YTSeg-CC-BY"
    default_tasks: tuple[str, ...] = ("asr", "achap")
    default_splits: tuple[str, ...] = ("train",)
    default_languages: tuple[str, ...] = ("en",)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare the YTSeg (CC BY) corpus.")
    add_common_args(
        parser,
        default_tasks=YTSegCCByCorpusPreparer.default_tasks,
        default_languages=YTSegCCByCorpusPreparer.default_languages,
        default_splits=YTSegCCByCorpusPreparer.default_splits,
    )
    args, extra_args = parser.parse_known_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    preparer = YTSegCCByCorpusPreparer(
        dataset_root=args.dataset_root,
        tasks=args.tasks,
        splits=args.splits,
        languages=args.languages
    )
    preparer.additional_setting(args=extra_args)
    preparer.prepare(
        skip_parquet=args.skip_parquet,
        skip_download=args.skip_download,
        skip_filter=args.skip_filter,
        args=args,
    )


if __name__ == "__main__":
    main()

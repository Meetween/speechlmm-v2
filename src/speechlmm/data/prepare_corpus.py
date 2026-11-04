"""Corpus preparation dispatcher.

    python -m speechlmm.data.prepare_corpus [--corpora NAME ...] [common args]

Forwards remaining arguments to :func:`speechlmm.data.corpus_preparer.main`, which
selects registered preparers, resolves versions from ``versions.json`` (or defaults),
and runs the versioned ``CorpusPreparer`` pipeline.

Common arguments include ``--dataset-root``, ``--dataset-version``, ``--version-file``,
``--tasks``, ``--splits``, ``--languages``, ``--skip-parquet``, ``--skip-download``,
and ``--skip-filter``. Corpus-specific flags are forwarded via ``additional_setting``.
"""

from __future__ import annotations

from speechlmm.data.corpus_preparer import main as corpus_preparer_main


def main(argv: list[str] | None = None) -> None:
    corpus_preparer_main(argv)


if __name__ == "__main__":
    main()

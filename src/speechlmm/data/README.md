# Data preparation

The pipeline converts raw corpora into the shared metadata schema, prepares media,
exports ShareGPT JSON, and writes a LLaMA-Factory `dataset_info.json`.

```text
raw annotations -> corpus Parquet conversion -> shared schema Parquet
                                                    |
                                                    +-> media preparation
                                                    |
                                                    +-> optional duration/task/corrupted-video filter
                                                    |
                                                    +-> prompt rendering -> ShareGPT JSON
                                                                            |
                                                                            +-> dataset_info.json
```

Supported corpora:

- `ami`
- `common_voice`
- `dgt_tm`
- `elitr`
- `europarl_st`
- `fleurs`
- `icsi`
- `lipcrops`
- `librisqa`
- `slurp`
- `speechmassive`
- `spokensquad`
- `topical_chat`
- `voxpopuli_eurospeech`
- `ytseg`
- `ytseg_cc_by`

## Prepare a corpus

Set `DATASETS_DIR`, or set `SCRATCH_DIR` to use `$SCRATCH_DIR/data`.

```bash
python -m speechlmm.data.prepare_corpus --corpus <name> [corpus options]
```

Use `--skip-parquet` to reuse metadata, `--skip-download` to reuse media, and
`--skip-filter` to keep the unfiltered Parquet. Run the command with `--help` after
the corpus name to see corpus-specific options.

By default, samples longer than 12 minutes (audio and video duration) are moved to
`rejected_parquet/`; the pipeline then builds ShareGPT from `filtered_parquet/` and
writes JSON under `filtered_sharegpt/`. Corpus-specific limits (for example AMI
avspeakd at 120 seconds) are set in the corpus preparer.

Samples whose `video_input_path` is missing or corrupted are moved to
`rejected_parquet/` too; each distinct file is decoded once (first and last frame).
Paths are resolved against `$DATASETS_DIR` (or `--dataset-root` / `--media-root`).
If none of the referenced videos exist the filter aborts, since that points at a
wrong root rather than at bad media. Pass `--no-check-video-integrity` to skip the
check entirely, or `--deep-video-check` to decode every frame instead, which catches
corruption in the middle of a file but is as slow as playing the whole video.

ShareGPT files and registry keys use
`<corpus>_<task>_<split>_<input-language>`.

## Layout

```text
data/
├── schema.py                 # Shared Parquet schema
├── utils.py                  # Shared task and media helpers
├── corpus_preparer.py        # Preparation pipeline
├── filter_parquet.py         # Optional duration/task/corrupted-video filter
├── video_integrity.py        # Corrupted video detection
├── prepare_corpus.py         # Corpus CLI dispatcher
├── parquet_to_sharegpt.py    # Parquet to ShareGPT conversion
├── prompt_renderer.py        # Prompt selection and rendering
├── prompts/                  # Versioned prompt templates
├── registry_dataset.py       # dataset_info.json registration
├── merge_sharegpt.py         # Optional physical JSON merge
├── corpus_stats.py           # Parquet statistics
├── preprocess_dataset.py     # LLaMA-Factory tokenization
└── processing/
    └── <corpus>/
        ├── to_parquet.py      # Corpus-specific metadata conversion
        ├── download.py        # Optional media preparation
        └── prepare.py         # CorpusPreparer subclass and CLI
```

## Prompts

`prompt_renderer.py` loads and validates templates from `prompts/`. The current
default is `v0_1.json`. Templates define instruction wording; the renderer supplies
languages, modalities, questions, answer options, intents, and matching media
placeholders. A sample can select a specialized template through
`metadata.prompt_variant`.

## Build a combined registry

For a training mix, prefer one registry that references the existing ShareGPT files:

```bash
python -m speechlmm.data.registry_dataset \
  --from-dir <shared-dataset-root>
```

Add `--filter` to register JSON under `filtered_sharegpt/` instead of `sharegpt/`.

If a consumer requires one physical JSON file, concatenate ShareGPT arrays explicitly:

```bash
python -m speechlmm.data.merge_sharegpt \
  --common-dir <parent-dir> \
  --output-file <output.json> \
  --input-files <sharegpt-dir/json-1.json> <sharegpt-dir/json-2.json> [...]
```

The merge utility loads all inputs into memory. Use the combined registry for large
training mixes to avoid copying the samples.

## Report corpus statistics

```bash
python -m speechlmm.data.corpus_stats --corpus <name>
python -m speechlmm.data.corpus_stats --corpus <name> --markdown
python -m speechlmm.data.corpus_stats --corpus <name> --filter
```

## Pre-tokenize a training dataset

```bash
python -m speechlmm.data.preprocess_dataset <config.yaml> [run_name=...]
```

## Add a corpus

1. Create `processing/<corpus>/` with an empty `__init__.py`.
2. Add `to_parquet.py`. Build rows with `empty_row()`, store media paths
   relative to the dataset root, cast to the shared `SCHEMA`, and write Parquet.
3. Add `download.py` if media must be downloaded, linked, or generated.
4. Add `prepare.py` with a `CorpusPreparer` subclass. Set `name`,
   `default_tasks`, `default_splits`, and `default_languages`; implement
   `to_parquet()`, override `download()` when needed, and expose `main(argv)`.

The default layout is one `<split>.parquet` per split. Override
`parquet_files_for_split()` when a logical split uses multiple Parquet files,
such as one file per language or translation direction.

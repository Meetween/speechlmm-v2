"""SpeechLMM-v2 official per-sample metadata schema.

Shared by the corpus-specific Parquet builders so the column set is defined once.
"""

from __future__ import annotations

import pyarrow as pa

# (column, pyarrow type). Order defines the Parquet column order.
SCHEMA_FIELDS: list[tuple[str, pa.DataType]] = [
    ("sample_id", pa.string()),
    ("task", pa.list_(pa.string())),
    ("split", pa.string()),
    ("source_dataset", pa.string()),
    ("license", pa.string()),
    ("audio_input_path", pa.string()),
    ("audio_input_duration", pa.float64()),
    ("audio_input_sample_rate", pa.int64()),
    ("audio_input_num_channels", pa.int64()),
    ("audio_input_codec", pa.string()),
    ("audio_output_path", pa.string()),
    ("audio_output_duration", pa.float64()),
    ("audio_output_sample_rate", pa.int64()),
    ("audio_output_num_channels", pa.int64()),
    ("audio_output_codec", pa.string()),
    ("audio_prompt_path", pa.string()),
    ("audio_prompt_duration", pa.float64()),
    ("audio_prompt_sample_rate", pa.int64()),
    ("text_input", pa.string()),
    ("text_input_normalized", pa.string()),
    ("num_words_input", pa.int64()),
    ("text_output", pa.string()),
    ("text_output_normalized", pa.string()),
    ("num_words_output", pa.int64()),
    ("text_prompt", pa.string()),
    ("language_audio_input", pa.string()),
    ("language_text_input", pa.string()),
    ("language_video_input", pa.string()),
    ("language_audio_output", pa.string()),
    ("language_text_output", pa.string()),
    ("language_audio_prompt", pa.string()),
    ("language_text_prompt", pa.string()),
    ("video_input_path", pa.string()),
    ("video_input_duration", pa.float64()),
    ("video_input_fps", pa.float64()),
    ("video_input_width", pa.int64()),
    ("video_input_height", pa.int64()),
    ("has_face", pa.bool_()),
    ("num_visible_speakers", pa.int64()),
    ("speaker_id", pa.list_(pa.string())),
    ("speaker_gender", pa.list_(pa.string())),
    ("speaker_age_group", pa.list_(pa.string())),
    ("num_speakers", pa.int64()),
    ("audio_input_snr_db", pa.float64()),
    ("audio_input_speech_ratio", pa.float64()),
    ("audio_input_clipped_ratio", pa.float64()),
    ("lid_audio_predicted", pa.string()),
    ("language_confidence_audio", pa.float64()),
    ("lid_text_predicted", pa.string()),
    ("language_confidence_text", pa.float64()),
    ("repetition_score", pa.float64()),
    ("words_per_second", pa.float64()),
    ("chars_per_second", pa.float64()),
    ("alignment_score", pa.float64()),
    ("wer_estimate", pa.float64()),
    ("cer_estimate", pa.float64()),
    ("text_duplicate_score", pa.float64()),
    ("text_duplicate_of", pa.string()),
    ("audio_duplicate_score", pa.float64()),
    ("audio_duplicate_of", pa.string()),
    ("has_overlapping_speech", pa.bool_()),
    (
        "metadata",
        pa.struct(
            [
                ("intents", pa.list_(pa.string())),
                ("options", pa.list_(pa.string())),
                ("prompt_variant", pa.string()),
            ]
        ),
    ),
]

SCHEMA = pa.schema(SCHEMA_FIELDS)
SCHEMA_COLUMNS = [name for name, _ in SCHEMA_FIELDS]


def empty_row() -> dict:
    """Return a schema row with every column set to None."""
    return {col: None for col in SCHEMA_COLUMNS}

import base64
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from typing import NamedTuple
import json

import requests
from openai import OpenAI
from vllm.assets.audio import AudioAsset
from vllm.utils.argparse_utils import FlexibleArgumentParser

SEED = 42

#SERVER_URL = "https://example.com/api"
#SERVER_NAME = "127.0.0.1"
#SERVER_PORT = "8091"
#SERVER_URL = f"{SERVER_NAME}:{SERVER_PORT}"
#print(f'SERVER_URL:{SERVER_URL}')
#MAX_WORKERS = 10




LANGUAGE_MAP: dict[str] = {
    "cs": "Czech",
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "hu": "Hungarian",
    "it": "Italian",
    "nl": "Dutch",
    "pt": "Portuguese",
    "sv": "Swedish"
}

DEFAULT_TASK_PROMPTS: dict[str, str] = {
    "asr": "Transcribe the following speech.",
    "st": "Translate the following speech from {lang_in} to {lang_out}.",
    "tts": "Read the following text aloud.",
    "s2st": "Translate the following speech from {lang_in} to {lang_out} and read the following text aloud.",
    "mt": "Translate the following text from {lang_in} to {lang_out}.",
    "slu": "Extract the intent and slots from the following speech.",
    "ssum": "Summarize the following speech.",
    "tsum": "Summarize the following text.",
    "sqa": "Answer the following question based on the audio.",
    "achap": "Identify chapter boundaries and titles in the following audio.",
    "lipread": "Transcribe the speech from the video.",
    "avspeakd": "Identify speakers and their speaking times in the video.",
    "ifol": "Execute",
    "dialog": "Continue the following dialog between a human and a virtual assistant, where the assistant can see the video and hear the audio inputs, and generate text, speech, or both as output depending on the context of the dialog.",
}


QUERY_TYPE_MAP: dict[str, str] = {
    "asr": "use_audio",
    "st": "use_audio",
    "tts": "text",
    "s2st": "use_audio",
    "mt": "text",
    "slu": "use_audio",
    "ssum": "use_audio",
    "tsum": "text",
    "sqa": "use_audio",
    "achap": "use_audio",
    "lipread": "use_video",
    "avspeakd": "use_mixed_modalities",
    "ifol": "use_mixed_modalities",
    "dialog": "use_mixed_modalities",
}


class QueryResult(NamedTuple):
    inputs: dict
    limit_mm_per_prompt: dict[str, int]


def encode_base64_content_from_url(content_url: str) -> str:
    """Encode a content retrieved from a remote url to base64 format."""

    with requests.get(content_url) as response:
        response.raise_for_status()
        result = base64.b64encode(response.content).decode("utf-8")

    return result


def encode_base64_content_from_file(file_path: str) -> str:
    """Encode a local file to base64 format."""
    with open(file_path, "rb") as f:
        content = f.read()
        result = base64.b64encode(content).decode("utf-8")
    return result


def get_video_url_from_path(video_path: str | None) -> str:
    """Convert a video path (local file or URL) to a video URL format for the API.

    If video_path is None or empty, returns the default URL.
    If video_path is a local file path, encodes it to base64 data URL.
    If video_path is a URL, returns it as-is.
    """
    if not video_path:
        # Default video URL
        return "https://huggingface.co/datasets/raushan-testing-hf/videos-test/resolve/main/sample_demo_1.mp4"

    # Check if it's a URL (starts with http:// or https://)
    if video_path.startswith(("http://", "https://")):
        return video_path

    # Otherwise, treat it as a local file path
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Detect video MIME type from file extension
    video_path_lower = video_path.lower()
    if video_path_lower.endswith(".mp4"):
        mime_type = "video/mp4"
    elif video_path_lower.endswith(".webm"):
        mime_type = "video/webm"
    elif video_path_lower.endswith(".mov"):
        mime_type = "video/quicktime"
    elif video_path_lower.endswith(".avi"):
        mime_type = "video/x-msvideo"
    elif video_path_lower.endswith(".mkv"):
        mime_type = "video/x-matroska"
    else:
        # Default to mp4 if extension is unknown
        mime_type = "video/mp4"

    video_base64 = encode_base64_content_from_file(video_path)
    return f"data:{mime_type};base64,{video_base64}"


def get_audio_url_from_path(audio_path: str | None) -> str:
    """Convert an audio path (local file or URL) to an audio URL format for the API.

    If audio_path is None or empty, returns the default URL.
    If audio_path is a local file path, encodes it to base64 data URL.
    If audio_path is a URL, returns it as-is.
    """
    if not audio_path:
        # Default audio URL
        return AudioAsset("mary_had_lamb").url

    # Check if it's a URL (starts with http:// or https://)
    if audio_path.startswith(("http://", "https://")):
        return audio_path

    # Otherwise, treat it as a local file path
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Detect audio MIME type from file extension
    audio_path_lower = audio_path.lower()
    if audio_path_lower.endswith((".mp3", ".mpeg")):
        mime_type = "audio/mpeg"
    elif audio_path_lower.endswith(".wav"):
        mime_type = "audio/wav"
    elif audio_path_lower.endswith(".ogg"):
        mime_type = "audio/ogg"
    elif audio_path_lower.endswith(".flac"):
        mime_type = "audio/flac"
    elif audio_path_lower.endswith(".m4a"):
        mime_type = "audio/mp4"
    else:
        # Default to wav if extension is unknown
        mime_type = "audio/wav"

    audio_base64 = encode_base64_content_from_file(audio_path)
    return f"data:{mime_type};base64,{audio_base64}"


def get_system_prompt():
    question = "Explain the system architecture for a scalable audio generation pipeline. Answer in 15 words."
    
    prompt = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": f"{question}",
            }
        ]
    }
    return prompt


default_system = (
    "You are SpeechLMM, a virtual human developed by the EU funded project Meetween, capable of perceiving auditory and visual inputs, as well as "
    "generating text and speech."
)


def get_query(custom_prompt, text_input: str | None = None, video_paths: list[str] | None = None, audio_paths: list[str] | None = None):
    """Query for use_audio_in_video mode.

    When use_audio_in_video=True, audio is automatically extracted from the video
    by the server. Do NOT send a separate audio_url - this would cause a mismatch
    between the number of audio and video items.

    # Note: audio is extracted from video automatically when use_audio_in_video=True
    # Do not include a separate audio_url here
    """
    question = custom_prompt

    if text_input:
        question = question + text_input

    content = []
    
    if video_paths:
        for i in range(len(video_paths)):
            video_url = get_video_url_from_path(video_paths[i])
            content.append({"type": "video_url", "video_url": {"url": video_url}})

    if audio_paths:
        for i in range(len(audio_paths)):
            audio_url = get_audio_url_from_path(audio_paths[i])
            content.append({"type": "audio_url", "audio_url": {"url": audio_url}})


    content.append({"type": "text", "text": question})
    
    return {
        "role": "user",
        "content": content
    }


def get_params() -> list[dict]:
    thinker_sampling_params = {
        "temperature": 0.4,  # Deterministic
        "top_p": 0.9,
        "top_k": 1,
        "max_tokens": 16384,
        "repetition_penalty": 1.05,
        "stop_token_ids": [151645],  # Qwen EOS token <|im_end|>
        "seed": SEED,
    }

    # Sampling parameters for Talker stage (codec generation)
    # Stop at codec EOS token
    talker_sampling_params = {
        "temperature": 0.9,
        "top_k": 50,
        "max_tokens": 4096,
        "seed": SEED,
        "detokenize": False,
        "repetition_penalty": 1.05,
        "stop_token_ids": [2150],  # TALKER_CODEC_EOS_TOKEN_ID
    }

    # # Sampling parameters for Code2Wav stage (audio generation)
    code2wav_sampling_params = {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "max_tokens": 4096 * 16,
        "seed": SEED,
        "detokenize": True,
        "repetition_penalty": 1.1,
    }

    sampling_params_list = [
        thinker_sampling_params,
        talker_sampling_params,
        code2wav_sampling_params,
    ]

    return sampling_params_list


def run_multimodal_generation(args, client: OpenAI) -> None:
    model_name = args.model
    input_json_file = args.input_json_file
    input_data_dir = args.input_data_dir
    output_json_file = args.output_json_file
    output_data_dir = args.output_data_dir
    output_relative_dir = args.output_relative_dir
    if output_json_file:
        os.makedirs(os.path.basename(output_json_file), exist_ok=True)
    if output_data_dir and output_relative_dir:
        os.makedirs(os.path.join(output_data_dir, output_relative_dir), exist_ok=True)
    modalities = args.modalities
    num_concurrent_requests = args.num_concurrent_requests

    print(f'run_multimodal_generation args: {args}')

    sampling_params_list = get_params()

    print(f'Reading input from {input_json_file}...')
    try:
        with open(input_json_file) as f:
            inputs_array = [json.loads(line.strip()) for line in f.readlines()]
    except json.JSONDecodeError:
        raise ValueError(f"Line {i} in {input_json_file} is not a valid JSON object: {line.strip()}")

    prompts_array, output_modalities_array, extra_body_array = [], [], []
    for i, obj in enumerate(inputs_array):
        # TODO: should we use text_input_normalized instead of text_input
        text_input = obj.get("text_input", None)
        id = obj.get("sample_id", None)

        task = obj.get("task", None)
        task = task[0] if task is not None and isinstance(task, list) and len(task) > 0 else task
        assert task is not None, f"Task field is required in the input JSON but not found in line {i}: {line.strip()}"
        assert task in QUERY_TYPE_MAP, f"Task {task} in line {i} is not supported. Supported tasks are: {list(QUERY_TYPE_MAP.keys())}"

        src_lng = obj.get("language_text_input", None)
        tgt_lng = obj.get("language_text_output", None)

        # Map language codes to full names if present in LANGUAGE_MAP, otherwise keep as-is
        src_lng = LANGUAGE_MAP[src_lng] if src_lng is not None and src_lng in LANGUAGE_MAP else src_lng
        tgt_lng = LANGUAGE_MAP[tgt_lng] if tgt_lng is not None and tgt_lng in LANGUAGE_MAP else tgt_lng

        video_path = obj.get("video_input_path", None)
        video_path = os.path.join(input_data_dir, video_path) if video_path and not video_path.startswith(("http://", "https://")) else video_path
        assert video_path is None or os.path.exists(video_path), f"Video file not found: {video_path}"

        audio_path = obj.get("audio_input_path", None)
        audio_path = os.path.join(input_data_dir, audio_path) if audio_path and not audio_path.startswith(("http://", "https://")) else audio_path
        assert audio_path is None or os.path.exists(audio_path), f"Audio file not found: {audio_path}"
        
        # get the default prompt for the task, and format it with the source and target languages if applicable
        task_prompt = DEFAULT_TASK_PROMPTS.get(task, "Process the following input.")
        task_prompt = task_prompt.format(lang_in=src_lng, lang_out=tgt_lng)

        # select the right query_type based on the task or other fields in the input JSON, or fallback to the command-line argument
        query_type = QUERY_TYPE_MAP[task]

        if query_type == "text":
            prompt = get_query(custom_prompt=task_prompt, text_input=text_input)
        elif query_type == "use_video":
            prompt = get_query(custom_prompt=task_prompt, video_paths=[video_path])
        elif query_type == "use_audio":
            prompt = get_query(custom_prompt=task_prompt, audio_paths=[audio_path])
        elif query_type == "use_audio_in_video":
            # Note: audio is extracted from video automatically when use_audio_in_video=True
            # Do not include a separate audio_url here
            prompt = get_query(custom_prompt=task_prompt, video_paths=[video_path])
        elif query_type == "use_mixed_modalities":
            prompt = get_query(custom_prompt=task_prompt, video_paths=[video_path], audio_paths=[audio_path])
        else:
            raise ValueError(f"Query type {query_type} is not a valid query type. Please check the input JSON or update the query_type mapping logic.")

        extra_body = {
            "sampling_params_list": sampling_params_list  # Optional, it has a default setting in stage_configs of the corresponding model.
            }

        if query_type == "use_audio_in_video":
            extra_body["mm_processor_kwargs"] = {"use_audio_in_video": True}

        if modalities is not None:
            output_modalities = modalities.split(",")
        else:
            output_modalities = None

        prompts_array.append(prompt)
        output_modalities_array.append(output_modalities)
        extra_body_array.append(extra_body)

    assert len(inputs_array) == len(prompts_array) == len(output_modalities_array) == len(extra_body_array), "Number of input arrays does not match"

    # Test multiple concurrent completions
    print(f'num_concurrent_requests: {num_concurrent_requests} number of prompts: {len(prompts_array)}')

    source_dataset = inputs_array[0].get("source_dataset", "output") if len(inputs_array) > 0 else "output"
    output_json_file = f"{source_dataset}.jsonl" if args.output_json_file is None else args.output_json_file
    with ThreadPoolExecutor(max_workers=num_concurrent_requests) as executor, open(output_file_path, "w") as outf:
        futures = []
        for i, (prompt, output_modalities, extra_body) in enumerate(zip(prompts_array, output_modalities_array, extra_body_array)):
            # Submit multiple completion requests concurrently
            futures.append(
                executor.submit(
                    client.chat.completions.create,
                    messages=[prompt,],
                    model=model_name,
                    modalities=output_modalities,
                    extra_body=extra_body,
                    stream=args.stream,
                )
            )
        
        print(f"Submitted all requests {len(prompts_array)}")
         
        i = 0
        # 👇 risultati appena pronti
        for future in as_completed(futures):
            chat_completion = future.result()

            for choice in chat_completion.choices:
                count = int(i / 2)

                if count >= len(prompts_array):
                    break

                if choice.message.audio:
                    audio_data = base64.b64decode(choice.message.audio.data)
                    sample_id = inputs_array[count].get("sample_id", count)  # Use sample_id from input JSON if available, otherwise fallback to count

                    # TODO : should we further organize the output audio files into subdirectories based on the output_language orther fields in the input JSON
                    audio_file_path = f"{sample_id}.wav"
                    audio_file_relative_path = os.path.join(output_relative_dir, audio_file_path) if output_relative_dir else audio_file_path
                    audio_file_absolute_path = os.path.join(output_data_dir, audio_file_relative_path) if output_data_dir else audio_file_relative_path
                    with open(audio_file_path, "wb") as f:
                        f.write(audio_data)

                    inputs_array[count]["generated_audio_path"] = audio_file_relative_path  # Add generated audio path to the output JSON
                if choice.message.content:
                    inputs_array[count]["generated_text"] = choice.message.content  # Add generated text to the output JSON
                i += 1

            # wait until both generated_audio_path and generated_text are ready
            outf.write(json.dumps(inputs_array[count]) + "\n")  # Write the original input JSON content to the output file, and we will add generated content to it while keeping the same format as input
            outf.flush()

    print(f"all requests completed and results saved to {output_file_path}")


def parse_args():
    parser = FlexibleArgumentParser(description="Demo on using vLLM for offline inference with multi-task multi-modal large language models")

    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        help="Model Name / Path",
    )
    parser.add_argument(
        "--input-json-file",
        "-i",
        type=str,
        default=None,
        help="Path to JSON input file.",
    )
    parser.add_argument(
        "--input-data-dir",
        "-id",
        type=str,
        default=None,
        help="Path to input data directory. This correspond to the base PATH for the relative paths specified in the input JSON file, e.g. for video_input_path and audio_input_path.",
    )
    parser.add_argument(
        "--output-json-file",
        "-o",
        type=str,
        default=None,
        help="Path to output JSON file.",
    )
    parser.add_argument(
        "--output-data-dir",
        "-od",
        type=str,
        default=None,
        help="Path to the output data base directory.",
    )
    parser.add_argument(
        "--output-relative-dir",
        "-pd",
        type=str,
        default=None,
        help="Relative path of output data directory with respect to the output database directory. Actual AV generated AV file will be stored into OUTPUT_DATA_DIR/output_relative_dir.",
    )
    parser.add_argument(
        "--modalities",
        type=str,
        default=None,
        help="Output modalities to use for the prompts.",
    )
    parser.add_argument(
        "--num-concurrent-requests",
        type=int,
        default=1,
        help="Number of concurrent requests to send. Default is 1.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8091,
        help="Port of the vLLM Omni API server.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host/IP of the vLLM Omni API server.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    host = args.host
    port = args.port
    openai_api_base = f"http://{host}:{port}/v1"
    client = OpenAI(
        api_key="EMPTY",
        base_url=openai_api_base,
    )
    print(f'client: {client}')
    run_multimodal_generation(args, client)

import torch
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForTextToWaveform,
)
import speechlmm.models


from torchcodec.decoders import VideoDecoder
from torchcodec.samplers import clips_at_regular_timestamps
from torchcodec.transforms import Resize
from torchvision.transforms import v2

import json
import csv
import os
import sys
import glob
from dataclasses import dataclass, field


@dataclass
class LipreadProcessor:
    H: int = 88
    W: int = 88
    transforms: v2.Compose = field(
        default_factory=lambda: v2.Compose(
            [
                v2.Grayscale(),
                v2.ToTensor(),
                v2.ToDtype(torch.float, scale=True),
                v2.Normalize(mean=[0.421], std=[0.165]),
            ]
        )
    )

    def __call__(self, video_path):
        video_dec = VideoDecoder(video_path, transforms=[Resize((self.H, self.W))])
        video = clips_at_regular_timestamps(
            video_dec, seconds_between_clip_starts=1 / 25
        )
        video = self.transforms(video.data)  # [t, b, 1, H, W]
        return video[:, 0, :, :]


# ==========================================
# 1. Load the Model and Tokenizer
# ==========================================
# Using a lightweight, capable model for easy execution
# model_id = "./llamafactory/saves/dev2/checkpoint-10"

# model_id = "./llamafactory/saves/lrs2bbc_train/adapter_only/checkpoint-2000"
# model_id = "./llamafactory/saves/LipCrops26052026/adapter_only"
# model_id = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
# model_id = "llamafactory/saves/lrs2bbc_train/stage2/"
#
model_id = sys.argv[3] if len(sys.argv) > 3 else ""

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id)

print("Loading model (this might take a moment)...")
# model = AutoModelForTextToWaveform.from_pretrained(
#     model_id, torch_dtype="auto", device_map="auto"
# )  # Automatically handles GPU/CPU allocation
model = AutoModel.from_pretrained(
    model_id, torch_dtype="bfloat16", device_map="auto"
)  # Automatically handles GPU/CPU allocation
# model.lipread_adapter.load_state_dict(torch.load("lipread_adapter.pth"))


# model.lipread_adapte

model.load_auto_avsr_weights("llamafactory/vsr.pth")
processor = LipreadProcessor()


# ==========================================
# 2. Custom Processing Method
# ==========================================
def custom_processing_method(message, data_folder):
    """
    Your custom logic goes here.
    You can inspect, alter, or inject messages into the conversation.
    """

    video = processor(data_folder + "/" + message["lipread"][0])

    # Example: Clean up user whitespace or do custom text modifications

    video_len = video.shape[0]
    video_str = (
        # "<|audio_start|>" + "<|lipread_pad|>" * (video_len // 1) + "<|audio_end|>"
        "<|lipread_start|>" + "<|lipread_pad|>" * (video_len // 1) + "<|lipread_end|>"
    )

    message_hystory = []
    message_hystory.append(message["messages"][0])
    message_hystory[0]["content"] = message_hystory[0]["content"].replace(
        "<lipread>", video_str
    )

    return message_hystory, video


# ==========================================
# 3. Read messages from JSON and run inference
# ==========================================
input_path = sys.argv[1]
output_file = sys.argv[2] if len(sys.argv) > 2 else "vsr_results.tsv"

with open(output_file, "w", newline="") as f:
    csv.writer(f, delimiter="\t").writerow(
        ["video_file", "ground_truth", "generated_text"]
    )

if os.path.isdir(input_path):
    # Folder mode: iterate over all .mp4 files in the directory
    data_folder = input_path
    video_files = sorted(glob.glob(os.path.join(data_folder, "*.mp4")))
    messages = []
    for vf in video_files:
        messages.append(
            {
                "lipread": [os.path.basename(vf)],
                "messages": [
                    {
                        "role": "user",
                        "content": "<lipread>Transcribe the content of the video",
                    }
                ],
            }
        )
else:
    data_folder = "./llamafactory/data"
    with open(input_path) as f:
        messages = json.load(f)

results = []
for i, message in enumerate(messages):
    video_path = message["lipread"][0]
    video_name = video_path

    print(f"\n[{i + 1}/{len(messages)}] Processing {video_name}...")

    processed_conversation, video = custom_processing_method(message, data_folder)

    model_inputs = tokenizer.apply_chat_template(
        processed_conversation, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    model_inputs["lipread"] = video.to("cuda").permute(1, 0, 2, 3)
    max_length = video.shape[0]
    lipread_mask = torch.ones(
        (len(model_inputs["lipread"]), max_length, max_length), dtype=torch.uint8
    )
    model_inputs["lipread_mask"] = lipread_mask

    model_inputs = {
        k: v.to(torch.bfloat16) if torch.is_floating_point(v) else v
        for k, v in model_inputs.items()
    }

    print("Generating response...")
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=100,
        # do_sample=True,
        # temperature=0.1,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )

    prompt_length = model_inputs["input_ids"].shape[0]
    new_tokens = generated_ids[0][prompt_length:]
    response = (
        tokenizer.decode(new_tokens, skip_special_tokens=True)
        .split("assistant\n")[1]
        .strip("\n")
    )

    ground_truth = (
        message["messages"][1]["content"] if len(message["messages"]) > 1 else ""
    )

    print(f"Response: {response}")
    results.append([video_name, ground_truth, response])

    with open(output_file, "a", newline="") as f:
        csv.writer(f, delimiter="\t").writerow([video_name, ground_truth, response])

print(f"\nResults saved to {output_file}")

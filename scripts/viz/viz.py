import argparse
import csv
import requests
from PIL import Image
import torch
import transformers
from transformers import AutoProcessor, LlavaForConditionalGeneration
from vl_saliency import Saliency
from vl_saliency.select import regex

parser = argparse.ArgumentParser()
parser.add_argument("csv_file", help="CSV file with columns: word, prompt, response, image_url")
parser.add_argument("--output_dir", default="figs", help="Directory to save figures")
args = parser.parse_args()

models_to_run = [
    ("base", "llava-hf/llava-1.5-7b-hf"),
    ("finetuned", "/users/teilers/scratch/saliency-alignment/models/saliency_llava-1.5-7b_kl_w0.5_1708610")
]

transformers.utils.logging.set_verbosity_error()

device = "cuda" if torch.cuda.is_available() else "cpu"

with open(args.csv_file, newline="") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

for model_type, model_path in models_to_run:
    print(f"Loading {model_type} model...")
    model = LlavaForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.float32,
        attn_implementation="eager",
    ).to(device)

    processor = AutoProcessor.from_pretrained(model_path, padding_side="left")

    for row in rows:
        word = row["word"]
        prompt = row["prompt"]
        response = row["response"]
        image_url = row["image_url"]

        print(f"\nProcessing word='{word}', image_url='{image_url}'")

        image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": image_url},
                    {"type": "text", "text": prompt},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": response}]},
        ]

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(device)

        with Saliency(model, backend="torch_eager"):
            out = model(**inputs)

        sal = out.saliency.view(
            image=image, processor=processor, input_ids=inputs.input_ids
        )

        try:
            fig = sal.plot(regex(word), alpha=0.83, cmap="inferno", title=f"Saliency Map for `{word}` ({model_type})")
            fig.savefig(f"{args.output_dir}/saliency_map_{model_type}_{word}.png")
            print(f"  Saved saliency map for '{word}' ({model_type})")
        except Exception as e:
            print(f"  Skipping word '{word}' ({model_type}) — could not find it in the tokens.")
            print(f"  Available tokens: {sal.decoded_gen_tokens}")
            print(f"  Error: {e}")

    del model
    del processor
    torch.cuda.empty_cache()

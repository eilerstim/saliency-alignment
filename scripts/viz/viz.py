import requests
from PIL import Image
import torch
import transformers
from transformers import AutoProcessor, LlavaForConditionalGeneration
from vl_saliency import Saliency
from vl_saliency.select import regex

word = "plants"

models_to_run = [
    ("base", "llava-hf/llava-1.5-7b-hf"),
    ("finetuned", "/users/teilers/scratch/saliency-alignment/models/saliency_llava-1.5-7b_kl_w0.5_1708610")
]

prompt = "What is shown in this image?"

# image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
# response = "The image shows a large, fluffy, and furry cat walking on a snow-covered ground."

# image_url = "https://production-livingdocs-bluewin-ch.imgix.net/2025/06/27/bb47ca9e-cd82-42f1-a0be-2167a833a505.jpeg?w=994&auto=format"
# response = "The image shows three Swiss fighter jets flying in formation against a clear blue sky."

# prompt = "How many cats are in this image?"
# image_url = "https://preview.redd.it/bored-get-a-cat-we-have-149-cats-to-choose-from-at-stray-v0-h6ahmpscjq1g1.png?width=1536&format=png&auto=webp&s=93b9667943d5b7be14a4047c1f1e40b20fee8c9d"
# response = "There are four cats in this image."

# prompt = "How many dots are in this image?"
# image_url = "https://assets.magazin.com/assets/img/0013/00136559/0013655993-01.jpg/garderobenhaken-dots.jpg?profile=pdsmain_1500"
# response = "There are five dots in this image."

image_url = "https://umbrellacreative.com.au/wp-content/uploads/2020/01/hide-the-pain-harold-why-you-should-not-use-stock-photos.jpg"
response = "The image shows an older man holding a cup of coffee and smiling awkwardly. He sits at a table with a laptop in front of him, and there are two plants in the background."

image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

transformers.utils.logging.set_verbosity_error()

device = "cuda" if torch.cuda.is_available() else "cpu"

for model_type, model_path in models_to_run:
    print(f"Running {model_type} model...")
    model = LlavaForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.float32,  # Requires float32 for loss computation
        attn_implementation="eager",  # to get attention weights
    ).to(device)

    processor = AutoProcessor.from_pretrained(model_path, padding_side="left")

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

    sal = out.saliency.view(  # Create a view and pass in context for regex/visualization
        image=image, processor=processor, input_ids=inputs.input_ids
    )
    print(sal.decoded_gen_tokens)

    fig = sal.plot(regex(word), alpha=0.83, cmap="inferno", title=f"Saliency Map for `{word}` ({model_type})")

    # save fig
    fig.savefig(f"/users/teilers/scratch/saliency-alignment/figs/saliency_map_{model_type}_{word}.png")
    
    del model
    del processor
    torch.cuda.empty_cache()


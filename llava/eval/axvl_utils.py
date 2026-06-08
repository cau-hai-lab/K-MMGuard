import math
import os

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor


def split_list(lst, n):
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def load_axvl_model(model_path, dtype="bfloat16", device_map="auto"):
    dtype_map = {
        "auto": "auto",
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    torch_dtype = dtype_map.get(dtype.lower())
    if torch_dtype is None:
        raise ValueError(f"Unsupported dtype: {dtype}")

    if device_map == "auto" and not torch.cuda.is_available():
        device_map = "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map=device_map,
    ).eval()
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    return model, processor


def load_rgb_image(path):
    return Image.open(path).convert("RGB")


def resolve_image_path(image_folder, image_file, mode=None):
    if mode in ["normal", "text"]:
        return os.path.join(image_folder, "SD", image_file)
    if mode:
        return os.path.join(image_folder, mode, image_file)
    return os.path.join(image_folder, image_file)


def _input_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def axvl_generate(
    model,
    processor,
    query,
    image=None,
    temperature=0.0,
    top_p=None,
    num_beams=1,
    max_new_tokens=1024,
):
    content = []
    images = None
    if image is not None:
        content.append({"type": "image"})
        images = [image]
    content.append({"type": "text", "text": query})

    messages = [
        {
            "role": "user",
            "content": content,
        }
    ]
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    processor_kwargs = {
        "text": [text],
        "padding": True,
        "return_tensors": "pt",
    }
    if images is not None:
        processor_kwargs["images"] = images

    inputs = processor(**processor_kwargs).to(_input_device())

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "num_beams": num_beams,
        "do_sample": temperature > 0,
    }
    if temperature > 0:
        gen_kwargs["temperature"] = temperature
        if top_p is not None:
            gen_kwargs["top_p"] = top_p

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **gen_kwargs)

    generated = output_ids[:, inputs.input_ids.shape[1] :]
    return processor.batch_decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

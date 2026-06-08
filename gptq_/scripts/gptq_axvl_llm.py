import argparse
import inspect
import json
import os
import sys
import traceback
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gptq import GPTQ
from quant import Quantizer

def patch_transformers_parallel_interface():
    import transformers.modeling_utils as modeling_utils
    import transformers.integrations.tensor_parallel as tensor_parallel

    if getattr(modeling_utils, "ALL_PARALLEL_STYLES", None) is None:
        modeling_utils.ALL_PARALLEL_STYLES = tensor_parallel.ParallelInterface()
    if getattr(tensor_parallel, "ALL_PARALLEL_STYLES", None) is None:
        tensor_parallel.ALL_PARALLEL_STYLES = modeling_utils.ALL_PARALLEL_STYLES


EXCLUDE_PREFIXES = (
    "vision_tower",
    "multi_modal_projector",
    "projector",
    "vision",
    "visual",
    "image",
    "siglip",
    "clip",
)


def parse_args():
    parser = argparse.ArgumentParser(description="GPTQ quantization script for AXVL language model modules")
    parser.add_argument("--model-id", default="skt/A.X-4.0-VL-Light")
    parser.add_argument("--calib-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--nsamples", type=int, default=64)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--wbits", type=int, default=4)
    parser.add_argument("--groupsize", type=int, default=128)
    parser.add_argument("--percdamp", type=float, default=0.01)
    parser.add_argument("--blocksize", type=int, default=128)
    parser.add_argument("--act-order", action="store_true")
    parser.add_argument("--static-groups", action="store_true")
    parser.add_argument("--sym", action="store_true")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--quantize-lm-head", action="store_true")
    parser.add_argument("--max-blocks", type=int, default=None)
    parser.add_argument("--max-layers", type=int, default=None)
    parser.add_argument("--save-safetensors", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def get_dtype(dtype_name: str):
    if dtype_name == "bf16":
        return torch.bfloat16
    if dtype_name == "fp16":
        return torch.float16
    return torch.float32


def load_processor(model_id: str, local_files_only: bool = False):
    try:
        processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
    except Exception:
        import transformers.processing_utils as pu

        if not hasattr(pu, "_validate_images_text_input_order"):
            def _validate_images_text_input_order(images=None, text=None):
                return images, text

            pu._validate_images_text_input_order = _validate_images_text_input_order
        try:
            processor = AutoProcessor.from_pretrained(
                model_id,
                trust_remote_code=True,
                local_files_only=local_files_only,
            )
        except Exception:
            return None
    return processor


def select_target_linears(model: nn.Module, quantize_lm_head: bool):
    linears = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        if any(
            name == prefix or name.startswith(prefix + ".") or ("." + prefix + ".") in name
            for prefix in EXCLUDE_PREFIXES
        ):
            continue

        if name.startswith("lm_head") or ".lm_head" in name:
            if not quantize_lm_head:
                continue

        if not name.startswith("language_model."):
            continue

        linears.append((name, module))

    return linears


def get_module_by_name(root: nn.Module, name: str):
    if name == "":
        return root
    attrs = name.split(".")
    module = root
    for attr in attrs:
        module = getattr(module, attr)
    return module


def discover_blocks(model: nn.Module, target_linear_names):
    blocks = []
    if hasattr(model, "language_model"):
        lm = model.language_model
        if hasattr(lm, "model") and hasattr(lm.model, "layers"):
            layers = lm.model.layers
            if hasattr(layers, "__len__"):
                for idx, block in enumerate(layers):
                    blocks.append((f"language_model.model.layers.{idx}", block))

    if blocks:
        return blocks

    import re

    block_names = []
    for name in target_linear_names:
        match = re.match(r"(language_model(?:\.[^.]+)*\.layers\.\d+)(?:\.|$)", name)
        if match:
            block_names.append(match.group(1))

    block_names = sorted(set(block_names))
    for block_name in block_names:
        try:
            block = get_module_by_name(model, block_name)
            blocks.append((block_name, block))
        except Exception:
            continue

    return blocks


def load_calibration_texts(calib_jsonl: str, nsamples: int):
    texts = []
    with open(calib_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict) and "text" in data:
                texts.append(str(data["text"]))
            if len(texts) >= nsamples:
                break
    return texts


def move_inputs(inputs, device):
    if hasattr(inputs, "to"):
        try:
            return inputs.to(device)
        except Exception:
            return {k: v.to(device) for k, v in inputs.items()}
    return {k: v.to(device) for k, v in inputs.items()}


def register_hook(linear, gptq_obj):
    def hook(module, inp, out):
        if len(inp) == 0:
            return
        inp_tensor = inp[0]
        if isinstance(out, torch.Tensor):
            out_tensor = out
        elif isinstance(out, (tuple, list)) and len(out) > 0:
            out_tensor = out[0]
        else:
            return
        gptq_obj.add_batch(inp_tensor.detach(), out_tensor.detach())

    return linear.register_forward_hook(hook)


def configure_quantizer(quantizer, args):
    try:
        quantizer.configure(args.wbits, perchannel=True, sym=args.sym, mse=False)
    except Exception:
        try:
            quantizer.configure(args.wbits, perchannel=True, sym=args.sym)
        except Exception as exc:
            print("Quantizer.configure failed with signature:", inspect.signature(quantizer.configure))
            raise exc


def main():
    args = parse_args()
    print(f"Loading model {args.model_id} with dtype={args.dtype}, device_map={args.device_map}")
    patch_transformers_parallel_interface()

    config = AutoConfig.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    if args.attn_implementation:
        for nested in [config, getattr(config, "text_config", None), getattr(config, "llm_config", None)]:
            if nested is None:
                continue
            if isinstance(nested, dict):
                nested["_attn_implementation"] = args.attn_implementation
                nested["attn_implementation"] = args.attn_implementation
            else:
                nested._attn_implementation = args.attn_implementation
                nested.attn_implementation = args.attn_implementation

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        config=config,
        trust_remote_code=True,
        torch_dtype=get_dtype(args.dtype),
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
        local_files_only=args.local_files_only,
    ).eval()

    processor = load_processor(args.model_id, local_files_only=args.local_files_only)
    tokenizer = None
    try:
        tokenizer = processor.tokenizer
    except Exception:
        tokenizer = None

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_id,
            trust_remote_code=True,
            local_files_only=args.local_files_only,
        )

    if tokenizer.pad_token is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    target_linears = select_target_linears(model, args.quantize_lm_head)
    target_names = [name for name, _ in target_linears]
    print(f"Total target Linear modules: {len(target_names)}")
    print("First 30 target names:")
    for name in target_names[:30]:
        print(f"  {name}")

    if len(target_names) == 0:
        raise RuntimeError("No target Linear modules found for quantization under language_model.")

    blocks = discover_blocks(model, target_names)
    if len(blocks) == 0:
        raise RuntimeError("No transformer blocks discovered under language_model.*layers.*")

    print(f"Discovered {len(blocks)} layer blocks")
    if args.max_layers is not None:
        blocks = blocks[: args.max_layers]
        print(f"Limiting to first {args.max_layers} blocks via --max-layers")
    if args.max_blocks is not None:
        blocks = blocks[: args.max_blocks]
        print(f"Limiting to first {args.max_blocks} blocks via --max-blocks")

    if args.dry_run:
        print("Dry run enabled: skipping calibration and saving.")
        return

    texts = load_calibration_texts(args.calib_jsonl, args.nsamples)
    if len(texts) < args.nsamples:
        print(f"WARNING: Only found {len(texts)} calibration samples, expected {args.nsamples}.")
    print(f"Using {len(texts)} calibration samples.")

    input_device = model.get_input_embeddings().weight.device

    quantized_module_names = []
    num_quantized = 0

    for block_name, block in blocks:
        print(f"Quantizing block {block_name}")
        block_linears = []
        for name, module in block.named_modules():
            if isinstance(module, nn.Linear):
                full_name = f"{block_name}.{name}" if name else block_name
                if any(
                    full_name == prefix
                    or full_name.startswith(prefix + ".")
                    or ("." + prefix + ".") in full_name
                    for prefix in EXCLUDE_PREFIXES
                ):
                    continue
                if full_name.startswith("lm_head") or ".lm_head" in full_name:
                    if not args.quantize_lm_head:
                        continue
                block_linears.append((full_name, module))

        if len(block_linears) == 0:
            print(f"Skipping block {block_name} with no target Linear modules.")
            continue

        print(f"  Found {len(block_linears)} Linear modules in block {block_name}.")
        gptq_objs = {}
        hooks = []
        for name, linear in block_linears:
            gptq_obj = GPTQ(linear)
            gptq_objs[name] = gptq_obj
            hooks.append(register_hook(linear, gptq_obj))

        successful = 0
        for text in tqdm(texts, desc=f"Calibrating {block_name}"):
            try:
                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=args.seqlen,
                )
                inputs = move_inputs(inputs, input_device)
                with torch.no_grad():
                    model(**inputs)
                successful += 1
            except Exception:
                traceback.print_exc()
                continue

        for hook in hooks:
            hook.remove()

        if successful == 0:
            raise RuntimeError(f"No successful calibration forward passes for block {block_name}.")

        print(f"  Completed {successful} successful calibration passes for block {block_name}.")

        for name, gptq_obj in gptq_objs.items():
            print(f"    Quantizing module {name}")
            quantizer = Quantizer()
            configure_quantizer(quantizer, args)
            gptq_obj.quantizer = quantizer
            gptq_obj.fasterquant(
                blocksize=args.blocksize,
                percdamp=args.percdamp,
                groupsize=args.groupsize,
                actorder=args.act_order,
                static_groups=args.static_groups,
            )
            gptq_obj.free()
            quantized_module_names.append(name)
            num_quantized += 1

        torch.cuda.empty_cache()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving quantized model to {output_dir}")
    model.save_pretrained(output_dir, safe_serialization=args.save_safetensors)
    if processor is not None:
        processor.save_pretrained(output_dir)
    else:
        tokenizer.save_pretrained(output_dir)

    report = {
        "model_id": args.model_id,
        "output_dir": str(output_dir),
        "quantized_scope": "language_model only",
        "wbits": args.wbits,
        "groupsize": args.groupsize,
        "percdamp": args.percdamp,
        "blocksize": args.blocksize,
        "act_order": args.act_order,
        "static_groups": args.static_groups,
        "sym": args.sym,
        "num_quantized_modules": num_quantized,
        "quantized_modules": quantized_module_names,
        "skipped_reason": "vision_tower and multi_modal_projector are not quantized",
        "weight_storage_note": "This GPTQ implementation stores dequantized quantized weights back into the original Linear dtype. It is not packed int4 and will not provide int4 kernel speedup.",
    }
    with open(output_dir / "gptq_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("Saved gptq_report.json")


if __name__ == "__main__":
    main()

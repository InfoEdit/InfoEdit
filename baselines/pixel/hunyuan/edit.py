"""Run HunyuanImage-3.0-Instruct on one or more input images for image editing.

Install requirements first (per the official model card):
    pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
        --index-url https://download.pytorch.org/whl/cu128
    pip install transformers accelerate safetensors pillow
    # Optional, up to 3x faster MoE inference (first run ~10 min kernel compile):
    pip install flashinfer-python==0.5.0

The model is loaded via `AutoModelForCausalLM` with `trust_remote_code=True`
(image generation / editing is dispatched through the model's own
`generate_image` method). Up to 3 input images are supported.

Example:
    python hunyuan_image_edit/edit.py \
        --image input1.png --image input2.png \
        --prompt "基于图一的logo，参考图二中冰箱贴的材质，制作一个新的冰箱贴" \
        --output output_hunyuan_edit.png
"""

import argparse
import os

import torch
from transformers import AutoModelForCausalLM

MODEL_ID = "tencent/HunyuanImage-3.0-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HunyuanImage-3.0-Instruct image editing")
    parser.add_argument(
        "--image",
        action="append",
        required=True,
        help="Path to an input image. Pass up to 3 times for multi-image fusion.",
    )
    parser.add_argument("--prompt", required=True, help="Edit instruction.")
    parser.add_argument("--output", default="output_hunyuan_edit.png")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model_id",
        default=MODEL_ID,
        help=(
            "HF repo id or local path. For the distilled 8-step variant use "
            "'tencent/HunyuanImage-3.0-Instruct-Distil' and set --diff_infer_steps 8."
        ),
    )
    parser.add_argument(
        "--image_size",
        default="auto",
        help="Output resolution, e.g. 'auto' or '1280x768'.",
    )
    parser.add_argument(
        "--use_system_prompt",
        default="en_unified",
        help="System prompt preset (default: en_unified).",
    )
    parser.add_argument(
        "--bot_task",
        default="think_recaption",
        help="Task mode. 'think_recaption' enables reasoning + prompt rewrite.",
    )
    parser.add_argument("--diff_infer_steps", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument(
        "--no_align_image_size",
        action="store_true",
        help="Disable aligning output size to the input image size.",
    )
    parser.add_argument(
        "--moe_impl",
        choices=["eager", "flashinfer"],
        default="eager",
        help="MoE backend. 'flashinfer' needs flashinfer-python installed.",
    )
    parser.add_argument(
        "--attn_implementation",
        choices=["sdpa", "flash_attention_2", "eager"],
        default="sdpa",
    )
    parser.add_argument("--verbose", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if len(args.image) > 3:
        raise ValueError(
            f"HunyuanImage-3.0-Instruct supports up to 3 input images, got {len(args.image)}."
        )

    kwargs = dict(
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto",
        moe_impl=args.moe_impl,
        moe_drop_tokens=True,
    )

    print(f"loading {args.model_id} ...")
    model = AutoModelForCausalLM.from_pretrained(args.model_id, **kwargs)
    model.load_tokenizer(args.model_id)
    print("model loaded")

    with torch.inference_mode():
        cot_text, samples = model.generate_image(
            prompt=args.prompt,
            image=args.image,
            seed=args.seed,
            image_size=args.image_size,
            use_system_prompt=args.use_system_prompt,
            bot_task=args.bot_task,
            infer_align_image_size=not args.no_align_image_size,
            diff_infer_steps=args.diff_infer_steps,
            max_new_tokens=args.max_new_tokens,
            verbose=args.verbose,
        )

    if cot_text:
        print("\n--- chain of thought / recaption ---")
        print(cot_text)
        print("--- end ---\n")

    output_paths = []
    if len(samples) == 1:
        samples[0].save(args.output)
        output_paths.append(os.path.abspath(args.output))
        print("image saved at", output_paths[-1])
    else:
        base, ext = os.path.splitext(args.output)
        for i, img in enumerate(samples):
            path = f"{base}_{i}{ext}"
            img.save(path)
            output_paths.append(os.path.abspath(path))
            print("image saved at", output_paths[-1])


if __name__ == "__main__":
    main()

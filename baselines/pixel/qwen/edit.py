"""Run Qwen-Image-Edit-2511 on one or more input images.

Install the bleeding-edge diffusers first:
    pip install "git+https://github.com/huggingface/diffusers"
    pip install transformers accelerate safetensors pillow

Example:
    python qwen_image_edit/edit.py \
        --image input1.png --image input2.png \
        --prompt "The magician bear is on the left, the alchemist bear is on the right." \
        --output output_image_edit_2511.png
"""

import argparse
import os

import torch
from diffusers import QwenImageEditPlusPipeline
from PIL import Image

MODEL_ID = "Qwen/Qwen-Image-Edit-2511"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen-Image-Edit-2511 inference")
    parser.add_argument(
        "--image",
        action="append",
        required=True,
        help="Path to an input image. Pass multiple times for multi-image editing.",
    )
    parser.add_argument("--prompt", required=True, help="Edit instruction.")
    parser.add_argument(
        "--negative_prompt", default=" ", help="Negative prompt (default: single space)."
    )
    parser.add_argument("--output", default="output_image_edit_2511.png")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_inference_steps", type=int, default=40)
    parser.add_argument("--true_cfg_scale", type=float, default=4.0)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--num_images_per_prompt", type=int, default=1)
    parser.add_argument(
        "--dtype",
        choices=["bf16", "fp16"],
        default="bf16",
        help="Computation dtype. bf16 recommended (per HF model card).",
    )
    parser.add_argument(
        "--offload",
        choices=["none", "model", "sequential"],
        default="none",
        help=(
            "VRAM strategy. 'none' = all on GPU (needs ~48GB+). "
            "'model' = swap whole modules CPU<->GPU per step (recommended on a single 48GB card). "
            "'sequential' = swap per-submodule, slowest but smallest footprint."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    torch_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    pipeline = QwenImageEditPlusPipeline.from_pretrained(MODEL_ID, torch_dtype=torch_dtype)
    print("pipeline loaded")

    if args.offload == "model":
        pipeline.enable_model_cpu_offload()
    elif args.offload == "sequential":
        pipeline.enable_sequential_cpu_offload()
    else:
        pipeline.to("cuda")

    if hasattr(pipeline, "vae") and pipeline.vae is not None:
        pipeline.vae.enable_tiling()
        pipeline.vae.enable_slicing()

    pipeline.set_progress_bar_config(disable=None)

    images = [Image.open(p).convert("RGB") for p in args.image]

    inputs = {
        "image": images,
        "prompt": args.prompt,
        "generator": torch.manual_seed(args.seed),
        "true_cfg_scale": args.true_cfg_scale,
        "negative_prompt": args.negative_prompt,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "num_images_per_prompt": args.num_images_per_prompt,
    }

    with torch.inference_mode():
        output = pipeline(**inputs)

    out_images = output.images
    output_paths = []
    if len(out_images) == 1:
        out_images[0].save(args.output)
        output_paths.append(os.path.abspath(args.output))
        print("image saved at", output_paths[-1])
    else:
        base, ext = os.path.splitext(args.output)
        for i, img in enumerate(out_images):
            path = f"{base}_{i}{ext}"
            img.save(path)
            output_paths.append(os.path.abspath(path))
            print("image saved at", output_paths[-1])


if __name__ == "__main__":
    main()

"""Edit infographics using Volcengine Ark Seedream image-edit models.

Counterpart to pixel/gpt/edit.py (OpenAI gpt-image) and
pixel/gemini/edit.py (Gemini). Same input schema, same per-operation file
convention, same output file naming — only the backend differs.

Backend:
  Endpoint: https://ark.cn-beijing.volces.com/api/v3/images/generations
  Auth:     Authorization: Bearer ${ARK_API_KEY}
  Image edit is performed by sending POST /images/generations with both
  `prompt` and `image` (URL or base64 data URI). The same endpoint covers
  text-to-image and image-edit; image-edit is triggered by including `image`.

Modes:
  Only online (synchronous) is implemented in this first cut.
  Optional multi-process parallelism via --num_workers.

Tutorials referenced:
  https://www.volcengine.com/docs/82379/1824121   (含图编辑完整示例)
  https://www.volcengine.com/docs/82379/1541523   (API 参数手册)

Usage:
  python pixel/seedream/edit.py --operation aspect_ratio \\
      --input_file data/editing_prompts_html/v17.jsonl \\
      --output_dir edited_html_infographics_seedream/v17/doubao-seedream-5-0-lite \\
      --num_workers 8

Auth:
  Set ARK_API_KEY in the environment, or pass --api_key.
"""

import os
import json
import time
import base64
import argparse
import mimetypes
import multiprocessing as mp
from io import BytesIO
from datetime import datetime

import requests
from tqdm import tqdm
from PIL import Image
import wandb


ARK_ENDPOINT = "https://ark.cn-beijing.volces.com/api/v3/images/generations"

# Seedream 5.0 Lite model id on Volcengine Ark (released 2026-01-28).
# Override via --model if needed; the API parameter manual is the source of
# truth: https://www.volcengine.com/docs/82379/1541523
DEFAULT_MODEL = "doubao-seedream-5-0-260128"

# Seedream 5.0 requires output >= 3,686,400 pixels (~1920^2). "adaptive"
# scales to the input and will 400 on small infographics, so default to 2K.
DEFAULT_SIZE = "2K"

NUM_VARIANTS = 1

OPERATIONS = ("add", "delete", "swap_intra", "swap_inter", "text_expand", "aspect_ratio")

# Mirrors pixel/gpt/edit.py — for aspect_ratio prompts we map the
# target_orientation to a concrete Seedream size string. Each option exceeds
# Seedream 5.0's 3,686,400-pixel minimum, with a few diverse aspect ratios
# to vary outputs across multiple AR prompts on the same item.
SEEDREAM_SIZE_BY_ORIENTATION = {
    "landscape": ["2400x1600", "2560x1440", "2304x1728", "2880x1280"],  # 3:2, 16:9, 4:3, ~9:4
    "portrait":  ["1600x2400", "1440x2560", "1728x2304"],                # 2:3, 9:16, 3:4
    "square":    ["2048x2048"],
}


# =============================================================================
# Size helpers
# =============================================================================

def size_for_orientation(orientation: str, idx: int) -> str | None:
    if not orientation:
        return None
    options = SEEDREAM_SIZE_BY_ORIENTATION.get(orientation)
    if not options:
        return None
    return options[idx % len(options)]


# =============================================================================
# I/O helpers (kept identical to pixel/gpt/edit.py where possible)
# =============================================================================

def input_file_for(base: str, operation: str) -> str:
    stem, ext = os.path.splitext(base)
    return f"{stem}.{operation}{ext}" if ext else f"{base}.{operation}"


def output_dir_for(base: str, operation: str) -> str:
    return os.path.join(base, operation)


def load_prompts(file_path: str, prompt_index: int):
    """Load editing prompts. Same schema as the gpt/gemini variants:
       - new: 'generated_edit_prompts' list (+ optional 'target_orientations')
       - legacy: 'generated_edit_prompt' string
    For aspect_ratio prompts the orientation is mapped to a Seedream size
    string and stored as item['target_size']."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if not item.get("image_path"):
                continue

            prompts_list = item.get("generated_edit_prompts")
            if isinstance(prompts_list, list) and prompts_list:
                idx = prompt_index if prompt_index < len(prompts_list) else 0
                item["generated_edit_prompt"] = str(prompts_list[idx]).strip()
                target_orientations = item.get("target_orientations")
                if isinstance(target_orientations, list) and len(target_orientations) > idx:
                    orientation = str(target_orientations[idx]).strip()
                    target_size = size_for_orientation(orientation, idx)
                    if target_size:
                        item["target_size"] = target_size
            elif item.get("generated_edit_prompt"):
                item["generated_edit_prompt"] = str(item["generated_edit_prompt"]).strip()
            else:
                continue

            if not item["generated_edit_prompt"]:
                continue
            data.append(item)
    return data


def all_variants_exist(output_dir: str, item_id: str) -> bool:
    return all(
        os.path.exists(os.path.join(output_dir, f"{item_id}_edited_{v}.png"))
        for v in range(1, NUM_VARIANTS + 1)
    )


def resolve_size(item: dict, default_size: str) -> str:
    """Pick the size for one item: target_size (aspect_ratio op) or default."""
    return item.get("target_size") or default_size


def encode_image_data_uri(image_path: str) -> str:
    """Encode a local image as a base64 data URI, the format the Ark API
    accepts for the `image` field (alongside plain http(s) URLs)."""
    mime, _ = mimetypes.guess_type(image_path)
    if not mime:
        mime = "image/png"
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"


# =============================================================================
# Online (synchronous) mode
# =============================================================================

def _post_edit(api_key: str, model: str, prompt: str, image_data_uri: str,
               size: str, watermark: bool, response_format: str,
               seed: int | None, timeout: int = 180) -> dict:
    payload = {
        "model": model,
        "prompt": prompt,
        "image": image_data_uri,
        "size": size,
        "watermark": watermark,
        "response_format": response_format,
        "sequential_image_generation": "disabled",
    }
    if seed is not None:
        payload["seed"] = seed
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    resp = requests.post(ARK_ENDPOINT, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _decode_response(resp_json: dict) -> Image.Image:
    """OpenAI-compatible response shape: {"data":[{"url":...} | {"b64_json":...}]}"""
    data = resp_json.get("data") or []
    if not data:
        raise RuntimeError(f"Empty data field in response: {resp_json}")
    first = data[0]
    if first.get("b64_json"):
        return Image.open(BytesIO(base64.b64decode(first["b64_json"])))
    if first.get("url"):
        r = requests.get(first["url"], timeout=120)
        r.raise_for_status()
        return Image.open(BytesIO(r.content))
    raise RuntimeError(f"No url or b64_json in response item: {first}")


def process_edits(api_key: str, model: str, default_size: str,
                  watermark: bool, response_format: str, seed: int | None,
                  data_list: list, output_dir: str, worker_id: int = 0):
    os.makedirs(output_dir, exist_ok=True)
    max_retries = 3
    success_count = 0
    fail_count = 0

    desc = f"Worker-{worker_id}" if worker_id else "Editing Infographics (Seedream)"
    for item in tqdm(data_list, desc=desc, position=worker_id):
        item_id = str(item.get("id"))
        image_path = item.get("image_path")
        edit_prompt = item.get("generated_edit_prompt")

        if not os.path.exists(image_path):
            tqdm.write(f"[Warning] Image not found for ID {item_id}: {image_path}")
            fail_count += 1
            continue

        target_size = resolve_size(item, default_size)
        try:
            image_data_uri = encode_image_data_uri(image_path)
        except Exception as e:
            tqdm.write(f"[Error] Failed to encode {image_path}: {e}")
            fail_count += 1
            continue

        item_variant_success = 0
        for var_idx in range(1, NUM_VARIANTS + 1):
            save_path = os.path.join(output_dir, f"{item_id}_edited_{var_idx}.png")
            if os.path.exists(save_path):
                item_variant_success += 1
                continue

            for attempt in range(max_retries):
                try:
                    resp_json = _post_edit(
                        api_key=api_key, model=model, prompt=edit_prompt,
                        image_data_uri=image_data_uri, size=target_size,
                        watermark=watermark, response_format=response_format,
                        seed=seed,
                    )
                    img = _decode_response(resp_json)
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    img.save(save_path)
                    item_variant_success += 1
                    break
                except requests.HTTPError as e:
                    body = e.response.text[:500] if e.response is not None else ""
                    tqdm.write(f"[HTTPError] ID {item_id} (Variant {var_idx}, "
                               f"Attempt {attempt + 1}/{max_retries}): {e} | {body}")
                    time.sleep(2 * (attempt + 1))
                except Exception as e:
                    tqdm.write(f"[Error] ID {item_id} (Variant {var_idx}, "
                               f"Attempt {attempt + 1}/{max_retries}): {e}")
                    time.sleep(2)

            time.sleep(0.5)

        if item_variant_success == NUM_VARIANTS:
            success_count += 1
        else:
            fail_count += 1

    return success_count, fail_count


def _worker_entry(worker_id, api_key, model, default_size, watermark,
                  response_format, seed, data_shard, output_dir, result_queue):
    try:
        success, fail = process_edits(
            api_key=api_key, model=model, default_size=default_size,
            watermark=watermark, response_format=response_format, seed=seed,
            data_list=data_shard, output_dir=output_dir, worker_id=worker_id,
        )
        result_queue.put((worker_id, success, fail))
    except Exception as e:
        print(f"[Worker-{worker_id}] Fatal error: {e}")
        result_queue.put((worker_id, 0, len(data_shard)))


def run_parallel(num_workers, api_key, model, default_size, watermark,
                 response_format, seed, data, output_dir):
    shards = [[] for _ in range(num_workers)]
    for i, item in enumerate(data):
        shards[i % num_workers].append(item)

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    processes = []

    for wid in range(num_workers):
        if not shards[wid]:
            continue
        p = ctx.Process(
            target=_worker_entry,
            args=(wid, api_key, model, default_size, watermark, response_format,
                  seed, shards[wid], output_dir, result_queue),
        )
        p.start()
        processes.append(p)

    total_success = 0
    total_fail = 0
    for _ in processes:
        wid, s, f = result_queue.get()
        total_success += s
        total_fail += f

    for p in processes:
        p.join()

    return total_success, total_fail


# =============================================================================
# W&B
# =============================================================================

def upload_to_wandb(args, per_op_plan, total_tasks, success, fail):
    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config={
            "model": args.model,
            "size": args.size,
            "watermark": args.watermark,
            "response_format": args.response_format,
            "input_file": args.input_file,
            "output_dir": args.output_dir,
            "operations": list(per_op_plan.keys()),
            "input_files": [p["input_file"] for p in per_op_plan.values()],
            "output_dirs": [p["output_dir"] for p in per_op_plan.values()],
            "limit": args.limit,
            "num_workers": args.num_workers,
            "total_tasks": total_tasks,
            "num_variants": NUM_VARIANTS,
        },
    )

    run.log({
        "total": success + fail,
        "success": success,
        "failed": fail,
        "success_rate": success / (success + fail) if (success + fail) > 0 else 0,
    })

    artifact = wandb.Artifact(
        name="edited-infographics-seedream",
        type="dataset",
        description=f"Edited infographics via {args.model} ({success} ok, {fail} failed)",
    )
    for op, plan in per_op_plan.items():
        outd = plan["output_dir"]
        if os.path.isdir(outd):
            artifact.add_dir(os.path.abspath(outd), name=op)
    run.log_artifact(artifact)

    run.finish()
    print(f"W&B run uploaded: {run.url}")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Edit Infographics using Volcengine Ark Seedream models")
    parser.add_argument("--api_key", type=str, default=None,
                        help="Volcengine Ark API key. Falls back to ARK_API_KEY env var.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="Seedream model id, e.g. doubao-seedream-5-0-lite")
    parser.add_argument("--size", type=str, default=DEFAULT_SIZE,
                        help="Output size: 'adaptive', '2K', '4K', or WxH like '2048x2048'. "
                             "For aspect_ratio prompts the per-item target_size overrides this.")
    parser.add_argument("--watermark", action="store_true", default=False,
                        help="Embed Seedream watermark on outputs")
    parser.add_argument("--response_format", type=str, default="b64_json",
                        choices=["b64_json", "url"],
                        help="Ark response format. URL responses are fetched and saved.")
    parser.add_argument("--seed", type=int, default=None, help="Optional generation seed")
    parser.add_argument("--input_file", type=str, default="editing_prompts.jsonl",
                        help="Base path of the editing-prompt file. The actual file per operation "
                             "is derived as <stem>.<operation><ext>.")
    parser.add_argument("--output_dir", type=str, default="edited_infographics_seedream",
                        help="Base output directory; per-op outputs go to <output_dir>/<operation>/.")
    parser.add_argument("--operation", type=str, default="add",
                        choices=["add", "delete", "swap_intra", "swap_inter",
                                 "text_expand", "aspect_ratio", "all"],
                        help="Which editing operation to run; 'all' iterates every operation.")
    parser.add_argument("--prompt_index", type=int, default=0,
                        help="When 'generated_edit_prompts' is a list, use this index.")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N examples")
    parser.add_argument("--no_skip", action="store_true",
                        help="Re-generate even if all variants already exist on disk")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print edit prompts without calling the API")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Number of parallel subprocess workers")
    parser.add_argument("--wandb_project", type=str, default="infographic-editing", help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name")
    parser.add_argument("--no_wandb", action="store_true", help="Disable W&B logging")
    return parser.parse_args()


def main():
    args = parse_args()

    api_key = args.api_key or os.environ.get("ARK_API_KEY")
    if not args.dry_run and not api_key:
        print("Error: provide --api_key or set ARK_API_KEY")
        return

    operations = list(OPERATIONS) if args.operation == "all" else [args.operation]

    per_op_plan = {}
    for op in operations:
        input_file = input_file_for(args.input_file, op)
        output_dir = output_dir_for(args.output_dir, op)

        if not os.path.exists(input_file):
            print(f"[{op}] Input file not found: {input_file} — skipping")
            continue

        data = load_prompts(input_file, prompt_index=args.prompt_index)
        for item in data:
            item["operation"] = op

        if args.limit is not None and args.limit > 0:
            data = data[:args.limit]
            print(f"[{op}] Limited to first {args.limit} tasks")

        if not args.no_skip and os.path.isdir(output_dir):
            before = len(data)
            data = [d for d in data if not all_variants_exist(output_dir, str(d.get("id")))]
            if before - len(data) > 0:
                print(f"[{op}] Skipping {before - len(data)} items already in {output_dir}")

        per_op_plan[op] = {"input_file": input_file, "output_dir": output_dir, "data": data}

    total_tasks = sum(len(p["data"]) for p in per_op_plan.values())
    if total_tasks == 0:
        print("Nothing to do.")
        return

    print("-" * 50)
    print(f"Mode:        online (Seedream)")
    print(f"Model:       {args.model}")
    print(f"Size:        {args.size}")
    print(f"Watermark:   {args.watermark}")
    print(f"Response:    {args.response_format}")
    print(f"Operations:  {', '.join(per_op_plan.keys())}")
    print(f"Total Tasks: {total_tasks}")
    for op, plan in per_op_plan.items():
        print(f"  - [{op}] {len(plan['data'])} tasks | in={plan['input_file']} | out={plan['output_dir']}/")
    print(f"Variants:    {NUM_VARIANTS}")
    print(f"Num Workers: {args.num_workers}")
    print("-" * 50)

    if args.dry_run:
        for op, plan in per_op_plan.items():
            print(f"\n[Dry Run] operation={op} — first 3 edit prompts:\n")
            for item in plan["data"][:3]:
                size = resolve_size(item, args.size)
                print(f"--- ID {item.get('id')} | {item.get('image_path')} | size={size} ---")
                print(item.get("generated_edit_prompt", ""))
                print()
        return

    total_success = 0
    total_fail = 0
    for op, plan in per_op_plan.items():
        data = plan["data"]
        if not data:
            continue
        output_dir = plan["output_dir"]
        os.makedirs(output_dir, exist_ok=True)
        print(f"\n=== Processing operation: {op} ({len(data)} tasks) → {output_dir}/ ===")

        if args.num_workers and args.num_workers > 1:
            s, f = run_parallel(
                num_workers=args.num_workers, api_key=api_key, model=args.model,
                default_size=args.size, watermark=args.watermark,
                response_format=args.response_format, seed=args.seed,
                data=data, output_dir=output_dir,
            )
        else:
            s, f = process_edits(
                api_key=api_key, model=args.model, default_size=args.size,
                watermark=args.watermark, response_format=args.response_format,
                seed=args.seed, data_list=data, output_dir=output_dir, worker_id=0,
            )
        total_success += s
        total_fail += f

    print(f"\nSuccess: {total_success} / {total_tasks}")
    print(f"Failed:  {total_fail} / {total_tasks}")
    print("Edited images saved:")
    for op, plan in per_op_plan.items():
        print(f"  - [{op}] {plan['output_dir']}/")

    if not args.no_wandb:
        upload_to_wandb(args, per_op_plan, total_tasks, total_success, total_fail)


if __name__ == "__main__":
    main()

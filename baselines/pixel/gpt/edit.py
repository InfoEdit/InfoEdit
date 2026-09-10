"""Edit infographics using OpenAI gpt-image models.

This is the OpenAI counterpart to pixel/gemini/edit.py (which uses Gemini).
Same input schema, same per-operation file convention, same output file
naming — only the model backend differs.

Modes:
  --use_batch off (default): synchronous client.images.edit() with retries.
                             Optional multi-process parallelism (--num_workers).
  --use_batch on:            OpenAI Batch API (/v1/images/edits) — uploads
                             images via Files API, submits one batch per op,
                             polls every --batch_poll_interval seconds.

Auth:
  Uses OPENAI_API_KEY from env unless --api_key is given.

Usage examples:
  # online, single worker
  python pixel/gpt/edit.py --operation add \\
      --input_file data/editing_prompts_html/v17.jsonl \\
      --output_dir edited_html_infographics

  # online, parallel workers
  python pixel/gpt/edit.py --operation all --num_workers 4 ...

  # batch (one batch job per operation)
  python pixel/gpt/edit.py --use_batch --operation all ...
"""

import os
import json
import time
import base64
import argparse
import multiprocessing as mp
from io import BytesIO
from datetime import datetime

from tqdm import tqdm
from PIL import Image
from openai import OpenAI
import wandb


DEFAULT_MODEL = "gpt-image-2"
DEFAULT_QUALITY = "medium"

NUM_VARIANTS = 1

OPERATIONS = ("add", "delete", "swap_intra", "swap_inter", "text_expand", "aspect_ratio")

# ---- gpt-image-2 dimension constraints (from official docs) ----
GPT_IMAGE_2_MAX_EDGE = 3840
GPT_IMAGE_2_MIN_PIXELS = 655_360
GPT_IMAGE_2_MAX_PIXELS = 8_294_400
GPT_IMAGE_2_MAX_RATIO = 3.0

# gpt-image-1 / gpt-image-1.5: only these fixed presets are valid.
FIXED_PRESETS = [(1024, 1024), (1536, 1024), (1024, 1536)]
FIXED_BY_ORIENTATION = {
    "landscape": "1536x1024",
    "portrait":  "1024x1536",
    "square":    "1024x1024",
}

# gpt-image-2 supports custom sizes (16-multiples) — cycle through a few
# orientations so multiple aspect_ratio prompts on the same item produce
# diverse outputs (mirrors GEMINI_AR_BY_ORIENTATION in pixel/gemini/edit.py).
GPT_IMAGE_2_SIZE_BY_ORIENTATION = {
    "landscape": ["1536x1024", "1792x1024", "1408x1024", "2384x1024"],  # ~3:2, 7:4, 11:8, ~21:9
    "portrait":  ["1024x1536", "1024x1792", "1024x1408"],
    "square":    ["1024x1024"],
}

RUNNING_STATES = {"validating", "in_progress", "finalizing", "cancelling"}


# =============================================================================
# Size helpers
# =============================================================================

def _round_to_16(x: int) -> int:
    return max(16, int(round(x / 16)) * 16)


def _fit_gpt_image_2(w: int, h: int) -> tuple[int, int]:
    """Snap (w, h) to a size gpt-image-2 will accept (16-multiple, ratio≤3:1,
    pixel count in [655_360, 8_294_400], max edge ≤ 3840)."""
    ratio = max(w, h) / max(min(w, h), 1)
    if ratio > GPT_IMAGE_2_MAX_RATIO:
        if w >= h:
            w = int(round(h * GPT_IMAGE_2_MAX_RATIO))
        else:
            h = int(round(w * GPT_IMAGE_2_MAX_RATIO))

    long_edge = max(w, h)
    if long_edge > GPT_IMAGE_2_MAX_EDGE:
        scale = GPT_IMAGE_2_MAX_EDGE / long_edge
        w, h = int(round(w * scale)), int(round(h * scale))

    px = w * h
    if px < GPT_IMAGE_2_MIN_PIXELS:
        scale = (GPT_IMAGE_2_MIN_PIXELS / px) ** 0.5
        w, h = int(round(w * scale)), int(round(h * scale))
    elif px > GPT_IMAGE_2_MAX_PIXELS:
        scale = (GPT_IMAGE_2_MAX_PIXELS / px) ** 0.5
        w, h = int(round(w * scale)), int(round(h * scale))

    return _round_to_16(w), _round_to_16(h)


def _fit_fixed_preset(w: int, h: int) -> tuple[int, int]:
    """Pick the gpt-image-1/1.5 preset closest to the source aspect ratio."""
    src_ratio = w / h
    return min(FIXED_PRESETS, key=lambda p: abs(p[0] / p[1] - src_ratio))


def size_for_image(image_path: str, model: str) -> str:
    """Default `size` derived from the input image's dimensions."""
    with Image.open(image_path) as im:
        w, h = im.size
    if model == "gpt-image-2":
        out_w, out_h = _fit_gpt_image_2(w, h)
    else:
        out_w, out_h = _fit_fixed_preset(w, h)
    return f"{out_w}x{out_h}"


def size_for_orientation(orientation: str, idx: int, model: str) -> str | None:
    """Map a target_orientation (landscape/portrait/square) to an OpenAI size string."""
    if not orientation:
        return None
    if model == "gpt-image-2":
        options = GPT_IMAGE_2_SIZE_BY_ORIENTATION.get(orientation)
        if not options:
            return None
        return options[idx % len(options)]
    return FIXED_BY_ORIENTATION.get(orientation)


# =============================================================================
# I/O helpers (kept identical to pixel/gemini/edit.py where possible)
# =============================================================================

def input_file_for(base: str, operation: str) -> str:
    stem, ext = os.path.splitext(base)
    return f"{stem}.{operation}{ext}" if ext else f"{base}.{operation}"


def output_dir_for(base: str, operation: str) -> str:
    return os.path.join(base, operation)


def load_prompts(file_path: str, prompt_index: int, model: str):
    """Load editing prompts. Same schema as pixel/gemini/edit.py:
       - new: 'generated_edit_prompts' list (+ optional 'target_orientations')
       - legacy: 'generated_edit_prompt' string
    For aspect_ratio prompts the orientation is mapped to an OpenAI size string
    and stored as item['target_size']."""
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
                    target_size = size_for_orientation(orientation, idx, model)
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


def resolve_size(item: dict, model: str) -> str:
    """Pick the size for one item: target_size (aspect_ratio op) or input dims."""
    return item.get("target_size") or size_for_image(item["image_path"], model)


# =============================================================================
# Online (synchronous) mode
# =============================================================================

def process_edits(client: OpenAI, model: str, quality: str,
                  data_list: list, output_dir: str, worker_id: int = 0):
    os.makedirs(output_dir, exist_ok=True)
    max_retries = 3
    success_count = 0
    fail_count = 0

    desc = f"Worker-{worker_id}" if worker_id else "Editing Infographics (GPT)"
    for item in tqdm(data_list, desc=desc, position=worker_id):
        item_id = str(item.get("id"))
        image_path = item.get("image_path")
        edit_prompt = item.get("generated_edit_prompt")

        if not os.path.exists(image_path):
            tqdm.write(f"[Warning] Image not found for ID {item_id}: {image_path}")
            fail_count += 1
            continue

        target_size = resolve_size(item, model)

        item_variant_success = 0
        for var_idx in range(1, NUM_VARIANTS + 1):
            save_path = os.path.join(output_dir, f"{item_id}_edited_{var_idx}.png")
            if os.path.exists(save_path):
                item_variant_success += 1
                continue

            for attempt in range(max_retries):
                try:
                    with open(image_path, "rb") as f_img:
                        response = client.images.edit(
                            model=model,
                            image=f_img,
                            prompt=edit_prompt,
                            size=target_size,
                            quality=quality,
                            n=1,
                        )
                    b64 = response.data[0].b64_json
                    img = Image.open(BytesIO(base64.b64decode(b64)))
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    img.save(save_path)
                    item_variant_success += 1
                    break
                except Exception as e:
                    tqdm.write(f"[Error] ID {item_id} (Variant {var_idx}, "
                               f"Attempt {attempt + 1}/{max_retries}): {e}")
                    time.sleep(2)

            time.sleep(1)

        if item_variant_success == NUM_VARIANTS:
            success_count += 1
        else:
            fail_count += 1

    return success_count, fail_count


def _worker_entry(worker_id, api_key, model, quality, data_shard,
                  output_dir, result_queue):
    try:
        client = OpenAI(api_key=api_key)
        success, fail = process_edits(
            client=client, model=model, quality=quality,
            data_list=data_shard, output_dir=output_dir, worker_id=worker_id,
        )
        result_queue.put((worker_id, success, fail))
    except Exception as e:
        print(f"[Worker-{worker_id}] Fatal error: {e}")
        result_queue.put((worker_id, 0, len(data_shard)))


def run_parallel(num_workers, api_key, model, quality, data, output_dir):
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
            args=(wid, api_key, model, quality, shards[wid], output_dir, result_queue),
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
# Batch mode (OpenAI Batch API)
# =============================================================================

def _upload_image_to_files(client: OpenAI, path: str) -> str:
    with open(path, "rb") as f:
        uploaded = client.files.create(file=f, purpose="vision")
    return uploaded.id


def process_edits_batch(client: OpenAI, model: str, quality: str,
                        data_list: list, output_dir: str,
                        poll_interval: int = 30):
    """Submit all edit requests as a single OpenAI batch job.

    Steps:
      1. For each item, upload its image via Files API to get a file_id.
      2. Write the JSONL of /v1/images/edits requests using `images:[{file_id}]`.
      3. Upload the JSONL (purpose='batch') and create the batch.
      4. Poll until terminal; download output_file_id; decode each row to
         {item_id}_edited_1.png. Errors are written to batch_errors_*.jsonl.

    Only variant 1 is produced in batch mode (matches NUM_VARIANTS=1 default).
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    input_jsonl = os.path.join(output_dir, f"batch_input_{timestamp}.jsonl")

    # ---- step 1+2: upload images, write JSONL ----
    print(f"[Batch] Uploading images and writing requests to {input_jsonl}")
    written = 0
    with open(input_jsonl, "w", encoding="utf-8") as f_out:
        for item in tqdm(data_list, desc="Upload+JSONL"):
            item_id = str(item.get("id"))
            image_path = item.get("image_path")
            edit_prompt = item.get("generated_edit_prompt")

            if not image_path or not edit_prompt:
                print(f"[Batch] Skipping ID {item_id}: missing image_path or prompt")
                continue
            if not os.path.exists(image_path):
                print(f"[Batch] Skipping ID {item_id}: local image not found: {image_path}")
                continue

            try:
                image_file_id = _upload_image_to_files(client, image_path)
            except Exception as e:
                print(f"[Batch] Failed to upload image for ID {item_id}: {e}")
                continue

            target_size = resolve_size(item, model)
            line = {
                "custom_id": f"item-{item_id}",
                "method": "POST",
                "url": "/v1/images/edits",
                "body": {
                    "model": model,
                    "images": [{"file_id": image_file_id}],
                    "prompt": edit_prompt,
                    "size": target_size,
                    "quality": quality,
                    "n": 1,
                },
            }
            f_out.write(json.dumps(line, ensure_ascii=False) + "\n")
            written += 1

    if written == 0:
        print("[Batch] No valid requests to submit.")
        return 0, len(data_list)

    # ---- step 3: upload JSONL, create batch ----
    print(f"[Batch] Uploading JSONL ({written} requests)")
    with open(input_jsonl, "rb") as f:
        input_file = client.files.create(file=f, purpose="batch")

    print(f"[Batch] Creating batch (model={model})")
    batch = client.batches.create(
        input_file_id=input_file.id,
        endpoint="/v1/images/edits",
        completion_window="24h",
        metadata={"description": "edit"},
    )
    print(f"[Batch] batch_id = {batch.id}")
    print(f"[Batch] (you can recover this batch later via the OpenAI dashboard "
          f"or the batches.retrieve API)")

    # ---- step 4: poll ----
    while batch.status in RUNNING_STATES:
        c = batch.request_counts
        print(f"[Batch] state={batch.status}; "
              f"completed={c.completed}/{c.total} failed={c.failed}; "
              f"sleeping {poll_interval}s...")
        time.sleep(poll_interval)
        batch = client.batches.retrieve(batch.id)

    print(f"[Batch] FINAL status={batch.status}")

    # ---- step 5: write any errors file ----
    if batch.error_file_id:
        err = client.files.content(batch.error_file_id).text
        err_path = os.path.join(output_dir, f"batch_errors_{timestamp}.jsonl")
        with open(err_path, "w", encoding="utf-8") as f:
            f.write(err)
        print(f"[Batch] Wrote partial errors to {err_path}")

    if batch.status != "completed" or not batch.output_file_id:
        print(f"[Batch] Batch did not complete cleanly: {batch.status}")
        return 0, len(data_list)

    # ---- step 6: parse responses, save PNGs ----
    raw = client.files.content(batch.output_file_id).text
    raw_path = os.path.join(output_dir, f"batch_output_{timestamp}.jsonl")
    with open(raw_path, "w", encoding="utf-8") as f:
        f.write(raw)

    success = 0
    fail = 0
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"[Batch] Skipping malformed line: {e}")
            fail += 1
            continue

        cid = rec.get("custom_id", "")
        item_id = cid[len("item-"):] if cid.startswith("item-") else cid

        if rec.get("error"):
            print(f"[Batch][Failed] ID {item_id}: {rec['error']}")
            fail += 1
            continue

        try:
            b64 = rec["response"]["body"]["data"][0]["b64_json"]
        except (KeyError, IndexError, TypeError):
            print(f"[Batch][Failed] ID {item_id}: unexpected response shape")
            fail += 1
            continue

        try:
            img = Image.open(BytesIO(base64.b64decode(b64)))
            if img.mode != "RGB":
                img = img.convert("RGB")
            save_path = os.path.join(output_dir, f"{item_id}_edited_1.png")
            img.save(save_path)
            success += 1
        except Exception as e:
            print(f"[Batch][Failed] ID {item_id}: decode/save error: {e}")
            fail += 1

    return success, fail


# =============================================================================
# W&B
# =============================================================================

def upload_to_wandb(args, per_op_plan, total_tasks, success, fail):
    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config={
            "model": args.model,
            "quality": args.quality,
            "input_file": args.input_file,
            "output_dir": args.output_dir,
            "operations": list(per_op_plan.keys()),
            "input_files": [p["input_file"] for p in per_op_plan.values()],
            "output_dirs": [p["output_dir"] for p in per_op_plan.values()],
            "limit": args.limit,
            "num_workers": args.num_workers,
            "use_batch": args.use_batch,
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
        name="edited-infographics-gpt",
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
    parser = argparse.ArgumentParser(description="Edit Infographics using OpenAI gpt-image models")
    parser.add_argument("--api_key", type=str, default=None,
                        help="OpenAI API key. Falls back to OPENAI_API_KEY env var.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        choices=["gpt-image-2", "gpt-image-1.5", "gpt-image-1"],
                        help="OpenAI image edit model")
    parser.add_argument("--quality", type=str, default=DEFAULT_QUALITY,
                        choices=["low", "medium", "high", "auto"],
                        help="Output image quality")
    parser.add_argument("--input_file", type=str, default="editing_prompts.jsonl",
                        help="Base path of the editing-prompt file. The actual file per operation "
                             "is derived as <stem>.<operation><ext>.")
    parser.add_argument("--output_dir", type=str, default="edited_infographics_gpt",
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
                        help="Number of parallel subprocess workers (online mode only)")
    parser.add_argument("--use_batch", action="store_true",
                        help="Use OpenAI Batch API. One batch job is submitted per operation. "
                             "Ignores --num_workers; produces only variant 1 per item.")
    parser.add_argument("--batch_poll_interval", type=int, default=30,
                        help="Seconds between batch job status polls")
    parser.add_argument("--wandb_project", type=str, default="infographic-editing", help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name")
    parser.add_argument("--no_wandb", action="store_true", help="Disable W&B logging")
    return parser.parse_args()


def main():
    args = parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not args.dry_run and not api_key:
        print("Error: provide --api_key or set OPENAI_API_KEY")
        return

    operations = list(OPERATIONS) if args.operation == "all" else [args.operation]

    per_op_plan = {}
    for op in operations:
        input_file = input_file_for(args.input_file, op)
        output_dir = output_dir_for(args.output_dir, op)

        if not os.path.exists(input_file):
            print(f"[{op}] Input file not found: {input_file} — skipping")
            continue

        data = load_prompts(input_file, prompt_index=args.prompt_index, model=args.model)
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
    print(f"Mode:        {'batch (OpenAI)' if args.use_batch else 'online'}")
    print(f"Model:       {args.model}")
    print(f"Quality:     {args.quality}")
    print(f"Operations:  {', '.join(per_op_plan.keys())}")
    print(f"Total Tasks: {total_tasks}")
    for op, plan in per_op_plan.items():
        print(f"  - [{op}] {len(plan['data'])} tasks | in={plan['input_file']} | out={plan['output_dir']}/")
    print(f"Variants:    {NUM_VARIANTS}")
    if args.use_batch:
        print(f"Poll Interval: {args.batch_poll_interval}s")
    else:
        print(f"Num Workers: {args.num_workers}")
    print("-" * 50)

    if args.dry_run:
        for op, plan in per_op_plan.items():
            print(f"\n[Dry Run] operation={op} — first 3 edit prompts:\n")
            for item in plan["data"][:3]:
                size = resolve_size(item, args.model)
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

        if args.use_batch:
            client = OpenAI(api_key=api_key)
            s, f = process_edits_batch(
                client=client, model=args.model, quality=args.quality,
                data_list=data, output_dir=output_dir,
                poll_interval=args.batch_poll_interval,
            )
        elif args.num_workers and args.num_workers > 1:
            s, f = run_parallel(
                num_workers=args.num_workers, api_key=api_key, model=args.model,
                quality=args.quality, data=data, output_dir=output_dir,
            )
        else:
            client = OpenAI(api_key=api_key)
            s, f = process_edits(
                client=client, model=args.model, quality=args.quality,
                data_list=data, output_dir=output_dir, worker_id=0,
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

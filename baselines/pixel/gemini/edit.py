import os
import json
import time
import base64
import random
import argparse
import multiprocessing as mp
from io import BytesIO
from datetime import datetime
from tqdm import tqdm
from PIL import Image
from google import genai
from google.genai import types
try:
    import wandb
except ImportError:   # optional: only needed when W&B logging is enabled
    wandb = None

DEFAULT_MODEL_PATH = "gemini-3.1-flash-image-preview" # "gemini-3-pro-image-preview"

NUM_VARIANTS = 1

OPERATIONS = ("add", "delete", "swap_intra", "swap_inter", "text_expand", "aspect_ratio")

# Gemini ImageConfig supports a fixed list of aspect-ratio strings; map the
# coarse target_orientation produced by generate_editing_prompts.py to a
# specific AR. Cycling by prompt_index keeps diversity across multiple prompts
# of the same orientation (e.g. landscape0 -> 3:2, landscape1 -> 16:9, ...).
GEMINI_AR_BY_ORIENTATION = {
    "landscape": ["3:2", "16:9", "4:3", "21:9"],
    "portrait":  ["2:3", "9:16", "3:4"],
    "square":    ["1:1"],
}


def gemini_ar_for_orientation(orientation: str, idx: int):
    options = GEMINI_AR_BY_ORIENTATION.get(orientation)
    if not options:
        return None
    return options[idx % len(options)]


def input_file_for(base: str, operation: str) -> str:
    stem, ext = os.path.splitext(base)
    return f"{stem}.{operation}{ext}" if ext else f"{base}.{operation}"


def output_dir_for(base: str, operation: str) -> str:
    return os.path.join(base, operation)


def load_prompts(file_path, prompt_index=0):
    """Load editing prompts. Accepts new schema (``generated_edit_prompts`` list)
    and legacy schema (``generated_edit_prompt`` string). For the list schema,
    picks ``prompt_index`` (default 0 = first prompt) and normalizes it onto
    ``generated_edit_prompt`` so downstream code stays unchanged.
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
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
                # Map the model-agnostic target_orientation to a Gemini AR string
                # so process_edits can pass it to Gemini's ImageConfig.
                target_orientations = item.get("target_orientations")
                if isinstance(target_orientations, list) and len(target_orientations) > idx:
                    orientation = str(target_orientations[idx]).strip()
                    ar = gemini_ar_for_orientation(orientation, idx)
                    if ar:
                        item["target_aspect_ratio"] = ar
            elif item.get("generated_edit_prompt"):
                item["generated_edit_prompt"] = str(item["generated_edit_prompt"]).strip()
            else:
                continue

            if not item["generated_edit_prompt"]:
                continue
            data.append(item)
    return data


def all_variants_exist(output_dir, item_id):
    return all(
        os.path.exists(os.path.join(output_dir, f"{item_id}_edited_{v}.png"))
        for v in range(1, NUM_VARIANTS + 1)
    )


def process_edits(clients, model_path, data_list, output_dir, worker_id=0):
    os.makedirs(output_dir, exist_ok=True)
    max_retries = 3
    success_count = 0
    fail_count = 0

    desc = f"Worker-{worker_id}" if worker_id else "Editing Infographics"
    for item in tqdm(data_list, desc=desc, position=worker_id):
        item_id = str(item.get("id"))
        image_path = item.get("image_path")
        edit_prompt = item.get("generated_edit_prompt")
        target_ar = item.get("target_aspect_ratio")  # only set for aspect_ratio op

        if not os.path.exists(image_path):
            tqdm.write(f"[Warning] Image not found for ID {item_id}: {image_path}")
            fail_count += 1
            continue

        try:
            source_image = Image.open(image_path)
        except Exception as e:
            tqdm.write(f"[Error] Failed to open image for ID {item_id}: {e}")
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
                    client = random.choice(clients)
                    config_kwargs = {
                        "thinking_config": types.ThinkingConfig(
                            thinking_level="High",
                            include_thoughts=True,
                        )
                    }
                    if target_ar:
                        config_kwargs["image_config"] = types.ImageConfig(aspect_ratio=target_ar)
                    response = client.models.generate_content(
                        model=model_path,
                        contents=[edit_prompt, source_image],
                        config=types.GenerateContentConfig(**config_kwargs),
                    )

                    generated_image = None
                    thoughts_text = []

                    parts_list = []
                    if hasattr(response, 'parts'):
                        parts_list = response.parts
                    elif hasattr(response, 'candidates') and response.candidates:
                        parts_list = response.candidates[0].content.parts

                    for part in parts_list:
                        if hasattr(part, 'text') and part.text:
                            thoughts_text.append(part.text)
                        if hasattr(part, 'inline_data') and part.inline_data is not None:
                            if not getattr(part, 'thought', False):
                                generated_image = Image.open(BytesIO(part.inline_data.data))

                    if generated_image:
                        txt_save_path = os.path.join(output_dir, f"{item_id}_edited_{var_idx}_thoughts.txt")
                        if generated_image.mode != 'RGB':
                            generated_image = generated_image.convert('RGB')
                        generated_image.save(save_path)

                        if thoughts_text:
                            with open(txt_save_path, 'w', encoding='utf-8') as f:
                                f.write("\n\n".join(thoughts_text))

                        item_variant_success += 1
                        break
                    else:
                        tqdm.write(f"[Warning] No edited image for ID {item_id} (Variant {var_idx}), retrying ({attempt + 1}/{max_retries})...")
                        time.sleep(2)

                except Exception as e:
                    tqdm.write(f"[Error] ID {item_id} (Variant {var_idx}, Attempt {attempt + 1}/{max_retries}): {e}")
                    time.sleep(2)

            time.sleep(1)

        if item_variant_success == NUM_VARIANTS:
            success_count += 1
        else:
            fail_count += 1

    return success_count, fail_count


def _worker_entry(worker_id, api_keys, model_path, data_shard, output_dir, result_queue):
    try:
        clients = [genai.Client(api_key=key, vertexai=True) for key in api_keys]
        success, fail = process_edits(
            clients=clients,
            model_path=model_path,
            data_list=data_shard,
            output_dir=output_dir,
            worker_id=worker_id,
        )
        result_queue.put((worker_id, success, fail))
    except Exception as e:
        print(f"[Worker-{worker_id}] Fatal error: {e}")
        result_queue.put((worker_id, 0, len(data_shard)))


def run_parallel(num_workers, api_keys, model_path, data, output_dir):
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
            args=(wid, api_keys, model_path, shards[wid], output_dir, result_queue),
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


def _batch_state_name(state) -> str:
    """Return a stable string state name regardless of SDK enum vs raw string."""
    return getattr(state, "name", None) or str(state)


def _guess_mime(path: str) -> str:
    lower = path.lower()
    if lower.endswith(".jpg") or lower.endswith(".jpeg"):
        return "image/jpeg"
    if lower.endswith(".webp"):
        return "image/webp"
    if lower.endswith(".gif"):
        return "image/gif"
    return "image/png"


def process_edits_batch(
    project_id,
    location,
    bucket_uri,
    model_path,
    data_list,
    output_dir,
    poll_interval=30,
):
    """Submit all edit requests as a single Vertex AI Gemini batch prediction job.

    Local images are uploaded to GCS under ``{bucket_uri}/batch_images_{ts}/`` so
    Vertex can read them via ``file_data.file_uri``. Responses are matched back to
    their source item by the ``(prompt_text, image_uri)`` tuple, which is unique
    per item.

    Auth uses Application Default Credentials — run `gcloud auth application-default login`
    beforehand. No API key is used in this path.

    Note: only variant 1 is produced in batch mode (output saved as
    ``{id}_edited_1.png``), matching the default ``NUM_VARIANTS=1``.
    """
    import fsspec  # lazy import — only needed in batch mode (requires gcsfs)
    from google.genai.types import CreateBatchJobConfig

    os.makedirs(output_dir, exist_ok=True)
    bucket_uri = bucket_uri.rstrip("/")
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    client = genai.Client(vertexai=True, project=project_id, location=location)
    fs = fsspec.filesystem("gcs")

    images_gcs_dir = f"{bucket_uri}/batch_images_{timestamp}"
    input_jsonl_uri = f"{bucket_uri}/batch_input_{timestamp}.jsonl"
    output_prefix = f"{bucket_uri}/batch_output_{timestamp}"

    items_by_key = {}  # (prompt_text, image_uri) -> (item_id, item)

    print(f"[Batch] Writing {len(data_list)} requests to {input_jsonl_uri}")
    written = 0
    with fs.open(input_jsonl_uri, "w", encoding="utf-8") as f_out:
        for item in data_list:
            item_id = str(item.get("id"))
            image_path = item.get("image_path")
            edit_prompt = item.get("generated_edit_prompt")
            target_ar = item.get("target_aspect_ratio")

            if not image_path or not edit_prompt:
                print(f"[Batch] Skipping ID {item_id}: missing image_path or prompt")
                continue

            if image_path.startswith("gs://"):
                image_uri = image_path
            else:
                if not os.path.exists(image_path):
                    print(f"[Batch] Skipping ID {item_id}: local image not found: {image_path}")
                    continue
                ext = os.path.splitext(image_path)[1] or ".png"
                image_uri = f"{images_gcs_dir}/{item_id}{ext}"
                try:
                    with open(image_path, "rb") as src, fs.open(image_uri, "wb") as dst:
                        dst.write(src.read())
                except Exception as e:
                    print(f"[Batch] Failed to upload image for ID {item_id}: {e}")
                    continue

            mime_type = _guess_mime(image_path)
            items_by_key[(edit_prompt, image_uri)] = (item_id, item)

            payload = {
                "request": {
                    "contents": [
                        {
                            "role": "user",
                            "parts": [
                                {"text": edit_prompt},
                                {"file_data": {"file_uri": image_uri, "mime_type": mime_type}},
                            ],
                        }
                    ],
                }
            }
            if target_ar:
                payload["request"]["generationConfig"] = {
                    "imageConfig": {"aspectRatio": target_ar}
                }
            f_out.write(json.dumps(payload, ensure_ascii=False) + "\n")
            written += 1

    if written == 0:
        print("[Batch] No valid requests to submit.")
        return 0, len(data_list)

    print(f"[Batch] Submitting batch job (model={model_path}, dest={output_prefix})")
    batch_job = client.batches.create(
        model=model_path,
        src=input_jsonl_uri,
        config=CreateBatchJobConfig(dest=output_prefix),
    )
    print(f"[Batch] Job name: {batch_job.name}")

    running_states = {"JOB_STATE_RUNNING", "JOB_STATE_PENDING", "JOB_STATE_QUEUED"}
    while _batch_state_name(batch_job.state) in running_states:
        print(f"[Batch] state={_batch_state_name(batch_job.state)}; sleeping {poll_interval}s...")
        time.sleep(poll_interval)
        batch_job = client.batches.get(name=batch_job.name)

    final_state = _batch_state_name(batch_job.state)
    if final_state != "JOB_STATE_SUCCEEDED":
        print(f"[Batch] Job {batch_job.name} failed: state={final_state} error={batch_job.error}")
        return 0, len(data_list)

    dest_uri = batch_job.dest.gcs_uri
    pred_paths = fs.glob(f"{dest_uri}/*/predictions.jsonl")
    if not pred_paths:
        print(f"[Batch] No predictions.jsonl found under {dest_uri}")
        return 0, len(data_list)

    success = 0
    fail = 0
    seen_keys = set()
    for pred_path in pred_paths:
        with fs.open(f"gs://{pred_path}", "r", encoding="utf-8") as f_in:
            for line in f_in:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"[Batch] Skipping malformed line: {e}")
                    fail += 1
                    continue

                prompt_text = None
                image_uri = None
                try:
                    parts = row["request"]["contents"][0]["parts"]
                    for p in parts:
                        if p.get("text"):
                            prompt_text = p["text"]
                        fd = p.get("file_data") or p.get("fileData")
                        if fd:
                            image_uri = fd.get("file_uri") or fd.get("fileUri")
                except (KeyError, IndexError, TypeError):
                    pass

                if not prompt_text or not image_uri:
                    print("[Batch] Skipping row without a recognizable request")
                    fail += 1
                    continue

                key = (prompt_text, image_uri)
                matched = items_by_key.get(key)
                if not matched:
                    print("[Batch] Could not match response back to any input item")
                    fail += 1
                    continue
                item_id, item = matched
                seen_keys.add(key)

                status = row.get("status", "") or ""
                response = row.get("response") or {}

                generated_image_b64 = None
                thoughts_text = []
                for cand in response.get("candidates", []) or []:
                    content = cand.get("content") or {}
                    for part in content.get("parts", []) or []:
                        if part.get("text"):
                            thoughts_text.append(part["text"])
                        inline_data = part.get("inlineData") or part.get("inline_data")
                        if inline_data and not part.get("thought"):
                            generated_image_b64 = inline_data.get("data")

                if not generated_image_b64:
                    print(f"[Batch][Failed] ID {item_id}: no image in response (status={status})")
                    fail += 1
                    continue

                try:
                    img_bytes = base64.b64decode(generated_image_b64)
                    generated_image = Image.open(BytesIO(img_bytes))
                    if generated_image.mode != "RGB":
                        generated_image = generated_image.convert("RGB")
                    save_path = os.path.join(output_dir, f"{item_id}_edited_1.png")
                    generated_image.save(save_path)
                    if thoughts_text:
                        txt_path = os.path.join(output_dir, f"{item_id}_edited_1_thoughts.txt")
                        with open(txt_path, "w", encoding="utf-8") as fo:
                            fo.write("\n\n".join(thoughts_text))
                    success += 1
                except Exception as e:
                    print(f"[Batch][Failed] ID {item_id}: decode/save error: {e}")
                    fail += 1

    missing = set(items_by_key) - seen_keys
    if missing:
        missing_ids = sorted({items_by_key[k][0] for k in missing})
        print(f"[Batch] {len(missing)} items had no row in predictions.jsonl: {missing_ids[:10]}...")
        fail += len(missing)

    return success, fail


def upload_to_wandb(args, per_op_plan, total_tasks, success, fail):
    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config={
            "model_path": args.model_path,
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
        name="edited-infographics",
        type="dataset",
        description=f"Edited infographics ({success} successful, {fail} failed)",
    )
    for op, plan in per_op_plan.items():
        outd = plan["output_dir"]
        if os.path.isdir(outd):
            artifact.add_dir(os.path.abspath(outd), name=op)
    run.log_artifact(artifact)

    run.finish()
    print(f"W&B run uploaded: {run.url}")


def parse_args():
    parser = argparse.ArgumentParser(description="Edit Infographics using Gemini Image Preview")
    parser.add_argument("--api_keys", type=str, nargs="+", default=None,
                        help="One or more Google API Keys (randomly selected per request). "
                             "Required for online mode; ignored when --use_batch is set.")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH, help="Gemini Image Model ID")
    parser.add_argument("--input_file", type=str, default="editing_prompts.jsonl",
                        help="Base path of the editing-prompt file. The actual file per operation is "
                             "derived as <stem>.<operation><ext> (e.g. editing_prompts.add.jsonl).")
    parser.add_argument("--output_dir", type=str, default="edited_infographics",
                        help="Base output directory. Edited images for each operation are written "
                             "to <output_dir>/<operation>/ (e.g. edited_infographics/add/).")
    parser.add_argument("--operation", type=str, default="add",
                        choices=["add", "delete", "swap_intra", "swap_inter", "text_expand", "aspect_ratio", "all"],
                        help="Which editing operation's prompts to run. 'all' iterates through every operation, "
                             "reading its per-op input file and writing to its per-op output subdir.")
    parser.add_argument("--prompt_index", type=int, default=0,
                        help="When the input file has a list under 'generated_edit_prompts', "
                             "use the prompt at this index (default 0 = first).")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N examples")
    parser.add_argument("--no_skip", action="store_true",
                        help="Re-generate even if all variants already exist on disk")
    parser.add_argument("--dry_run", action="store_true", help="Print edit prompts without calling the API")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Number of parallel subprocess workers")
    parser.add_argument("--use_batch", action="store_true",
                        help="Use Vertex AI Gemini Batch Prediction (async, GCS-staged). "
                             "Requires --gcp_project and --batch_bucket_uri; uses ADC auth (no api_key). "
                             "Ignores --api_keys / --num_workers. Produces only variant 1 per item.")
    parser.add_argument("--gcp_project", type=str, default=None,
                        help="GCP project ID for Vertex AI batch prediction (required with --use_batch)")
    parser.add_argument("--gcp_location", type=str, default="us-central1",
                        help="GCP location for Vertex AI batch prediction")
    parser.add_argument("--batch_bucket_uri", type=str, default=None,
                        help="Cloud Storage bucket URI (e.g. gs://my-bucket) used to stage batch input/output. "
                             "Bucket should be in --gcp_location (typically us-central1).")
    parser.add_argument("--batch_poll_interval", type=int, default=30,
                        help="Seconds between batch job status polls")
    parser.add_argument("--wandb_project", type=str, default="infographic-editing", help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name")
    parser.add_argument("--no_wandb", action="store_true", help="Disable W&B logging")
    return parser.parse_args()


def main():
    args = parse_args()

    operations = list(OPERATIONS) if args.operation == "all" else [args.operation]

    per_op_plan = {}  # op -> {"input_file", "output_dir", "data"}
    for op in operations:
        input_file = input_file_for(args.input_file, op)
        output_dir = output_dir_for(args.output_dir, op)

        if not os.path.exists(input_file):
            print(f"[{op}] Input file not found: {input_file} — skipping this operation")
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
                print(f"[{op}] Skipping {before - len(data)} items with all {NUM_VARIANTS} variants already in {output_dir}")

        per_op_plan[op] = {"input_file": input_file, "output_dir": output_dir, "data": data}

    total_tasks = sum(len(p["data"]) for p in per_op_plan.values())
    if total_tasks == 0:
        print("Nothing to do.")
        return

    print("-" * 50)
    print(f"Mode:        {'batch (Vertex AI)' if args.use_batch else 'online'}")
    print(f"Model:       {args.model_path}")
    print(f"Operations:  {', '.join(per_op_plan.keys())}")
    print(f"Total Tasks: {total_tasks}")
    for op, plan in per_op_plan.items():
        print(f"  - [{op}] {len(plan['data'])} tasks | in={plan['input_file']} | out={plan['output_dir']}/")
    print(f"Variants:    {NUM_VARIANTS}")
    if args.use_batch:
        print(f"GCP Project:   {args.gcp_project}")
        print(f"GCP Location:  {args.gcp_location}")
        print(f"Bucket URI:    {args.batch_bucket_uri}")
        print(f"Poll Interval: {args.batch_poll_interval}s")
    else:
        print(f"Num Workers: {args.num_workers}")
        print(f"API Keys:    {len(args.api_keys) if args.api_keys else 0}")
    print("-" * 50)

    if args.dry_run:
        for op, plan in per_op_plan.items():
            print(f"\n[Dry Run] operation={op} — first 3 edit prompts:\n")
            for item in plan["data"][:3]:
                print(f"--- ID {item.get('id')} | {item.get('image_path')} ---")
                print(item.get("generated_edit_prompt", ""))
                print()
        return

    if args.use_batch:
        if not args.gcp_project:
            print("Error: --use_batch requires --gcp_project")
            return
        if not args.batch_bucket_uri:
            print("Error: --use_batch requires --batch_bucket_uri (e.g. gs://my-bucket)")
            return
    elif not args.api_keys:
        print("Error: --api_keys is required")
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
            s, f = process_edits_batch(
                project_id=args.gcp_project,
                location=args.gcp_location,
                bucket_uri=args.batch_bucket_uri,
                model_path=args.model_path,
                data_list=data,
                output_dir=output_dir,
                poll_interval=args.batch_poll_interval,
            )
        elif args.num_workers and args.num_workers > 1:
            s, f = run_parallel(
                num_workers=args.num_workers,
                api_keys=args.api_keys,
                model_path=args.model_path,
                data=data,
                output_dir=output_dir,
            )
        else:
            clients = [genai.Client(api_key=key) for key in args.api_keys]
            s, f = process_edits(
                clients=clients,
                model_path=args.model_path,
                data_list=data,
                output_dir=output_dir,
                worker_id=0,
            )
        total_success += s
        total_fail += f

    print(f"\nSuccess: {total_success} / {total_tasks}")
    print(f"Failed:  {total_fail} / {total_tasks}")
    print("Edited images saved:")
    for op, plan in per_op_plan.items():
        print(f"  - [{op}] {plan['output_dir']}/")

    if not args.no_wandb and wandb is None:

        raise SystemExit("W&B logging requested but wandb is not installed — pip install wandb, or pass --no_wandb.")

    if not args.no_wandb:
        upload_to_wandb(args, per_op_plan, total_tasks, total_success, total_fail)


if __name__ == "__main__":
    main()

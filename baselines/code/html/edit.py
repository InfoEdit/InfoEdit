"""Code-based infographic editing.

Instead of editing the rendered PNG with an image-generation model
(``pixel/gemini/edit.py``), we hand the *unrendered HTML source* to an LLM,
ask it to apply the same editing instruction at the code level, save the
modified HTML, and finally render it to PNG with Playwright. Downstream
``evaluate_edits.py`` consumes the produced ``{id}_edited_1.png`` exactly as
in the image-edit pipeline.
"""

import os
import json
import time
import random
import argparse
import multiprocessing as mp
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from PIL import Image
try:                      # Vertex-only; unused on the openai-proxy branch
    from google import genai
    from google.genai import types
except ImportError:       # pragma: no cover
    genai = types = None
try:
    import wandb
except ImportError:   # optional: only needed when W&B logging is enabled
    wandb = None

# Reach llm_client.py at the repo root.
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[3]))
from llm_client import chat_text


DEFAULT_MODEL_PATH = "gemini-3.1-pro-preview"
NUM_VARIANTS = 1
OPERATIONS = ("add", "delete", "swap_intra", "swap_inter", "text_expand", "aspect_ratio")
INPUT_MODES = ("code", "code_image")

# generate_editing_prompts.py records target_orientations (landscape / portrait
# / square) instead of model-specific AR strings. The HTML reflow path is
# unconstrained by any model's supported-AR list, so we pick a representative
# AR per orientation for sizing the new canvas (compute_new_dims uses it).
HTML_AR_BY_ORIENTATION = {
    "landscape": ["3:2", "16:9", "4:3", "21:9"],
    "portrait":  ["2:3", "9:16", "3:4"],
    "square":    ["1:1"],
}


def html_ar_for_orientation(orientation: str, idx: int):
    options = HTML_AR_BY_ORIENTATION.get(orientation)
    if not options:
        return None
    return options[idx % len(options)]


def _guess_mime(path):
    lower = path.lower()
    if lower.endswith(".jpg") or lower.endswith(".jpeg"):
        return "image/jpeg"
    if lower.endswith(".webp"):
        return "image/webp"
    if lower.endswith(".gif"):
        return "image/gif"
    return "image/png"


def _thinking_config(model_path: str):
    """Build a ThinkingConfig appropriate for the model.

    Gemini 3.x accepts ``thinking_level`` ("High" etc.); Gemini 2.5 and earlier
    only accept ``thinking_budget`` and reject ``thinking_level``. We pass -1
    (dynamic budget, no cap) on 2.5 as the closest equivalent of "High" effort.
    """
    if model_path.startswith("gemini-3"):
        return types.ThinkingConfig(thinking_level="High", include_thoughts=True)
    return types.ThinkingConfig(thinking_budget=-1, include_thoughts=True)


def _thinking_config_json(model_path: str) -> dict:
    """JSON (camelCase) form of ``_thinking_config`` for batch payloads."""
    if model_path.startswith("gemini-3"):
        return {"thinkingLevel": "High", "includeThoughts": True}
    return {"thinkingBudget": -1, "includeThoughts": True}

EDIT_SYSTEM_INSTRUCTION = (
    "You are an expert front-end engineer who edits standalone HTML infographic posters. "
    "You will receive the full HTML source of an infographic and an editing instruction. "
    "Apply the instruction at the code level and return the COMPLETE modified HTML document.\n\n"
    "Hard requirements:\n"
    "  - The ouput HTML document MUST start with <!DOCTYPE html> or <html>.\n"
    "  - No prose, no explanations, no markdown code fences (no ```html ... ```).\n"
    "  - Preserve every piece of content, styling, and structure unrelated to the requested edit.\n"
    "  - Keep external assets (image URIs, fonts, scripts) referenced exactly as in the original "
    "unless the instruction explicitly demands changing them.\n"
    "  - When TARGET_DIMENSIONS_PX is provided (aspect-ratio edits), update the document's outer "
    "width/height (root container size, viewBox, body/html sizing) to match those dimensions exactly, "
    "and reflow the layout so it fills the new canvas without distortion or overflow.\n"
    "  - Otherwise preserve the original outer dimensions exactly."
)


def input_file_for(base, operation):
    stem, ext = os.path.splitext(base)
    return f"{stem}.{operation}{ext}" if ext else f"{base}.{operation}"


def output_dir_for(base, operation):
    return os.path.join(base, operation)


def html_path_for(image_path):
    return os.path.splitext(image_path)[0] + ".html"


def meta_path_for(image_path):
    return os.path.splitext(image_path)[0] + ".meta.json"


def load_prompts(file_path, prompt_index=0):
    """Mirror of ``pixel/gemini/edit.py: load_prompts``: handles list vs legacy schema
    and surfaces ``target_aspect_ratio`` for the aspect_ratio op."""
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
                    ar = html_ar_for_orientation(orientation, idx)
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


def parse_aspect_ratio(target_ar):
    if not target_ar or ":" not in target_ar:
        return None
    try:
        a, b = target_ar.split(":")
        a, b = float(a), float(b)
    except ValueError:
        return None
    if a <= 0 or b <= 0:
        return None
    return a, b


def compute_new_dims(orig_w, orig_h, target_ar):
    """Pick new (w, h) for an aspect-ratio edit by preserving total pixel area."""
    parsed = parse_aspect_ratio(target_ar)
    if not parsed:
        return orig_w, orig_h
    a, b = parsed
    area = orig_w * orig_h
    new_w = int(round((area * a / b) ** 0.5))
    new_h = int(round(new_w * b / a))
    return new_w, new_h


def load_meta(image_path):
    p = meta_path_for(image_path)
    if not os.path.exists(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def build_user_prompt(html_content, edit_prompt, target_ar=None, new_w=None, new_h=None, with_image=False):
    parts = [f"EDITING_INSTRUCTION:\n{edit_prompt}"]
    if target_ar and new_w and new_h:
        parts.append(
            f"TARGET_ASPECT_RATIO: {target_ar}\n"
            f"TARGET_DIMENSIONS_PX: width={new_w}, height={new_h}"
        )
    if with_image:
        parts.append(
            "RENDERED_PREVIEW: An image rendered from the ORIGINAL_HTML below is attached. "
            "Use it as visual reference for the current layout, colors, and component positions. "
            "All modifications, however, MUST be made in the HTML source — do not output an image."
        )
    parts.append(f"ORIGINAL_HTML:\n{html_content}")
    return "\n\n".join(parts)


def clean_html_output(text):
    """Strip stray markdown fences in case the model ignores the no-fence rule."""
    if not text:
        return ""
    text = text.strip()
    if text.startswith("```"):
        nl = text.find("\n")
        if nl != -1:
            text = text[nl + 1 :]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    return text.strip()


def edited_paths(output_dir, item_id, var_idx=1):
    base = os.path.join(output_dir, f"{item_id}_edited_{var_idx}")
    return base + ".html", base + ".png", base + "_thoughts.txt", base + ".render.json"


def all_pngs_exist(output_dir, item_id):
    return all(
        os.path.exists(edited_paths(output_dir, item_id, v)[1])
        for v in range(1, NUM_VARIANTS + 1)
    )


def all_htmls_exist(output_dir, item_id):
    return all(
        os.path.exists(edited_paths(output_dir, item_id, v)[0])
        for v in range(1, NUM_VARIANTS + 1)
    )


def process_edits(clients, model_path, data_list, output_dir, worker_id=0, input_mode="code"):
    """Online (synchronous) per-item HTML editing. Writes ``{id}_edited_1.html``
    and a ``.render.json`` sidecar with target rendering dimensions.

    When ``input_mode == "code_image"``, the rendered PNG referenced by
    ``image_path`` is attached as a multimodal input alongside the HTML so the
    LLM has visual context for the edit.
    """
    os.makedirs(output_dir, exist_ok=True)
    max_retries = 3
    success_count = 0
    fail_count = 0
    with_image = input_mode == "code_image"

    desc = f"Worker-{worker_id}" if worker_id else "Editing HTML"
    for item in tqdm(data_list, desc=desc, position=worker_id):
        item_id = str(item.get("id"))
        image_path = item.get("image_path")
        edit_prompt = item.get("generated_edit_prompt")
        target_ar = item.get("target_aspect_ratio")

        html_path = html_path_for(image_path)
        if not os.path.exists(html_path):
            tqdm.write(f"[Warning] HTML not found for ID {item_id}: {html_path}")
            fail_count += 1
            continue
        try:
            with open(html_path, "r", encoding="utf-8") as f:
                html_content = f.read()
        except OSError as e:
            tqdm.write(f"[Error] Failed to read HTML for ID {item_id}: {e}")
            fail_count += 1
            continue

        source_image = None
        if with_image:
            if not image_path or not os.path.exists(image_path):
                tqdm.write(f"[Warning] Rendered preview not found for ID {item_id}: {image_path}")
                fail_count += 1
                continue
            try:
                source_image = Image.open(image_path)
            except Exception as e:
                tqdm.write(f"[Error] Failed to open preview for ID {item_id}: {e}")
                fail_count += 1
                continue

        meta = load_meta(image_path) or {}
        orig_w = int(meta.get("width", 1200))
        orig_h = int(meta.get("height", 900))
        new_w, new_h = compute_new_dims(orig_w, orig_h, target_ar) if target_ar else (orig_w, orig_h)

        user_prompt = build_user_prompt(
            html_content, edit_prompt, target_ar, new_w, new_h, with_image=with_image,
        )

        item_variant_success = 0
        for var_idx in range(1, NUM_VARIANTS + 1):
            html_save, png_save, txt_save, render_save = edited_paths(output_dir, item_id, var_idx)
            if os.path.exists(html_save):
                item_variant_success += 1
                continue

            for attempt in range(max_retries):
                try:
                    edited_html = chat_text(
                        model_path,
                        user_prompt,
                        images=[source_image] if source_image is not None else None,
                        system=EDIT_SYSTEM_INSTRUCTION,
                    )
                    # The proxy exposes no separate reasoning stream, so there are
                    # no thoughts to save alongside the edit.
                    thoughts = []

                    edited_html = clean_html_output(edited_html)
                    if edited_html:
                        with open(html_save, "w", encoding="utf-8") as f:
                            f.write(edited_html)
                        with open(render_save, "w", encoding="utf-8") as f:
                            json.dump({"width": new_w, "height": new_h}, f)
                        if thoughts:
                            with open(txt_save, "w", encoding="utf-8") as f:
                                f.write("\n\n".join(thoughts))
                        item_variant_success += 1
                        break

                    tqdm.write(
                        f"[Warning] Empty HTML for ID {item_id} (Variant {var_idx}), "
                        f"retrying ({attempt + 1}/{max_retries})..."
                    )
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


def _worker_entry(worker_id, api_keys, model_path, data_shard, output_dir, result_queue, input_mode):
    try:
        clients = [None]  # unused: llm_client holds the proxy client
        success, fail = process_edits(
            clients=clients,
            model_path=model_path,
            data_list=data_shard,
            output_dir=output_dir,
            worker_id=worker_id,
            input_mode=input_mode,
        )
        result_queue.put((worker_id, success, fail))
    except Exception as e:
        print(f"[Worker-{worker_id}] Fatal error: {e}")
        result_queue.put((worker_id, 0, len(data_shard)))


def run_parallel(num_workers, api_keys, model_path, data, output_dir, input_mode="code"):
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
            args=(wid, api_keys, model_path, shards[wid], output_dir, result_queue, input_mode),
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


def _batch_state_name(state):
    return getattr(state, "name", None) or str(state)


def process_edits_batch(
    project_id,
    location,
    bucket_uri,
    model_path,
    data_list,
    output_dir,
    poll_interval=30,
    input_mode="code",
):
    """Submit all HTML-editing requests as a single Vertex AI Gemini batch job.

    The full prompt text (which embeds each item's unique HTML) is the matching
    key between the response row and the original item, so we don't need labels.

    When ``input_mode == "code_image"``, each item's rendered PNG is uploaded
    to GCS and attached to the request alongside the HTML text part.
    """
    import fsspec  # lazy — gcsfs only needed in batch mode
    from google.genai.types import CreateBatchJobConfig

    os.makedirs(output_dir, exist_ok=True)
    bucket_uri = bucket_uri.rstrip("/")
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    with_image = input_mode == "code_image"

    client = genai.Client(vertexai=True, project=project_id, location=location)
    fs = fsspec.filesystem("gcs")

    input_jsonl_uri = f"{bucket_uri}/code_batch_input_{timestamp}.jsonl"
    output_prefix = f"{bucket_uri}/code_batch_output_{timestamp}"
    images_gcs_dir = f"{bucket_uri}/code_batch_images_{timestamp}" if with_image else None

    prompt_to_item = {}      # full user prompt -> (item_id, new_w, new_h)
    print(f"[Batch] Writing {len(data_list)} requests to {input_jsonl_uri} (input_mode={input_mode})")
    written = 0
    with fs.open(input_jsonl_uri, "w", encoding="utf-8") as f_out:
        for item in data_list:
            item_id = str(item.get("id"))
            image_path = item.get("image_path")
            edit_prompt = item.get("generated_edit_prompt")
            target_ar = item.get("target_aspect_ratio")

            html_path = html_path_for(image_path)
            if not os.path.exists(html_path):
                print(f"[Batch] Skipping ID {item_id}: HTML not found at {html_path}")
                continue
            try:
                with open(html_path, "r", encoding="utf-8") as fh:
                    html_content = fh.read()
            except OSError as e:
                print(f"[Batch] Failed to read HTML for ID {item_id}: {e}")
                continue

            image_uri = None
            if with_image:
                if not image_path:
                    print(f"[Batch] Skipping ID {item_id}: missing image_path")
                    continue
                if image_path.startswith("gs://"):
                    image_uri = image_path
                else:
                    if not os.path.exists(image_path):
                        print(f"[Batch] Skipping ID {item_id}: rendered preview not found: {image_path}")
                        continue
                    ext = os.path.splitext(image_path)[1] or ".png"
                    image_uri = f"{images_gcs_dir}/{item_id}{ext}"
                    try:
                        with open(image_path, "rb") as src, fs.open(image_uri, "wb") as dst:
                            dst.write(src.read())
                    except Exception as e:
                        print(f"[Batch] Failed to upload preview for ID {item_id}: {e}")
                        continue

            meta = load_meta(image_path) or {}
            orig_w = int(meta.get("width", 1200))
            orig_h = int(meta.get("height", 900))
            new_w, new_h = compute_new_dims(orig_w, orig_h, target_ar) if target_ar else (orig_w, orig_h)

            user_prompt = build_user_prompt(
                html_content, edit_prompt, target_ar, new_w, new_h, with_image=with_image,
            )
            prompt_to_item[user_prompt] = (item_id, new_w, new_h)

            user_parts = [{"text": user_prompt}]
            if with_image and image_uri:
                user_parts.append({
                    "file_data": {"file_uri": image_uri, "mime_type": _guess_mime(image_path)}
                })

            payload = {
                "request": {
                    "systemInstruction": {"parts": [{"text": EDIT_SYSTEM_INSTRUCTION}]},
                    "contents": [
                        {"role": "user", "parts": user_parts}
                    ],
                    "generationConfig": {
                        "thinkingConfig": _thinking_config_json(model_path),
                    },
                }
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

    running = {"JOB_STATE_RUNNING", "JOB_STATE_PENDING", "JOB_STATE_QUEUED"}
    while _batch_state_name(batch_job.state) in running:
        print(f"[Batch] state={_batch_state_name(batch_job.state)}; sleeping {poll_interval}s")
        time.sleep(poll_interval)
        batch_job = client.batches.get(name=batch_job.name)

    final_state = _batch_state_name(batch_job.state)
    if final_state != "JOB_STATE_SUCCEEDED":
        print(f"[Batch] Job {batch_job.name} failed: state={final_state} error={batch_job.error}")
        return 0, len(data_list)

    dest_uri = batch_job.dest.gcs_uri
    pred_paths = fs.glob(f"{dest_uri}/*/predictions.jsonl")
    if not pred_paths:
        print(f"[Batch] No predictions.jsonl under {dest_uri}")
        return 0, len(data_list)

    success = 0
    fail = 0
    seen = set()
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
                try:
                    parts = row["request"]["contents"][0]["parts"]
                    for p in parts:
                        if p.get("text"):
                            prompt_text = p["text"]
                except (KeyError, IndexError, TypeError):
                    pass

                matched = prompt_to_item.get(prompt_text) if prompt_text else None
                if not matched:
                    print("[Batch] Could not match response back to any input item")
                    fail += 1
                    continue
                item_id, new_w, new_h = matched
                seen.add(prompt_text)

                status = row.get("status", "") or ""
                response = row.get("response") or {}
                edited_html = ""
                thoughts = []
                for cand in response.get("candidates", []) or []:
                    content = cand.get("content") or {}
                    for part in content.get("parts", []) or []:
                        text = part.get("text")
                        if not text:
                            continue
                        if part.get("thought"):
                            thoughts.append(text)
                        else:
                            edited_html += text

                edited_html = clean_html_output(edited_html)
                if not edited_html:
                    print(f"[Batch][Failed] ID {item_id}: empty HTML in response (status={status})")
                    fail += 1
                    continue

                html_save, _, txt_save, render_save = edited_paths(output_dir, item_id, 1)
                try:
                    with open(html_save, "w", encoding="utf-8") as fo:
                        fo.write(edited_html)
                    with open(render_save, "w", encoding="utf-8") as fo:
                        json.dump({"width": new_w, "height": new_h}, fo)
                    if thoughts:
                        with open(txt_save, "w", encoding="utf-8") as fo:
                            fo.write("\n\n".join(thoughts))
                    success += 1
                except OSError as e:
                    print(f"[Batch][Failed] ID {item_id}: write error: {e}")
                    fail += 1

    missing = set(prompt_to_item) - seen
    if missing:
        missing_ids = sorted({prompt_to_item[k][0] for k in missing})
        print(f"[Batch] {len(missing)} items had no row in predictions.jsonl: {missing_ids[:10]}...")
        fail += len(missing)

    return success, fail


def _render_one(html_path_str, png_path_str, width, height, wait_ms, max_attempts=5):
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return f"[SKIP] playwright not installed: {html_path_str}"

    last_err = None
    for attempt in range(1, max_attempts + 1):
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch()
                page = browser.new_page(
                    viewport={"width": width, "height": height},
                    device_scale_factor=1,
                )
                page.goto(Path(html_path_str).absolute().as_uri())
                page.wait_for_timeout(wait_ms)
                page.screenshot(
                    path=png_path_str,
                    full_page=False,
                    clip={"x": 0, "y": 0, "width": width, "height": height},
                )
                page.close()
                browser.close()
            suffix = "" if attempt == 1 else f" [attempt {attempt}/{max_attempts}]"
            return f"[OK] {Path(png_path_str).name} ({width}x{height}){suffix}"
        except Exception as e:
            last_err = e
            if attempt < max_attempts:
                time.sleep(1)
    return f"[ERR] {Path(html_path_str).name}: {last_err} (after {max_attempts} attempts)"


def render_all(output_dir, num_workers=8, wait_ms=2500, force=False):
    """Render every ``*_edited_*.html`` in ``output_dir`` to a sibling PNG using
    the dimensions stored in the matching ``.render.json`` sidecar."""
    out = Path(output_dir)
    if not out.is_dir():
        print(f"[Render] {output_dir} does not exist")
        return 0, 0
    html_files = sorted(out.glob("*_edited_*.html"))
    if not html_files:
        print(f"[Render] No HTML files in {output_dir}")
        return 0, 0

    tasks = []
    for h in html_files:
        png = h.with_suffix(".png")
        if png.exists() and not force:
            continue
        rj = h.with_suffix(".render.json")
        w, ht = 1200, 900
        if rj.exists():
            try:
                rd = json.loads(rj.read_text())
                w = int(rd.get("width", w))
                ht = int(rd.get("height", ht))
            except (OSError, json.JSONDecodeError):
                pass
        tasks.append((str(h), str(png), w, ht))

    if not tasks:
        print(f"[Render] All PNGs already exist in {output_dir}")
        return len(html_files), 0

    print(f"[Render] Rendering {len(tasks)} HTML files with {num_workers} workers")
    success = 0
    fail = 0
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        futs = {ex.submit(_render_one, h, p, w, ht, wait_ms): h for h, p, w, ht in tasks}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Rendering"):
            try:
                msg = fut.result()
            except Exception as e:
                print(f"[Render] Future exception: {e}")
                fail += 1
                continue
            if msg.startswith("[OK]"):
                success += 1
            else:
                print(msg)
                fail += 1
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
            "approach": "code-based-html-edit",
            "input_mode": args.input_mode,
        },
    )
    run.log({
        "total": success + fail,
        "success": success,
        "failed": fail,
        "success_rate": success / (success + fail) if (success + fail) > 0 else 0,
    })
    artifact = wandb.Artifact(
        name="edited-infographics-code",
        type="dataset",
        description=f"Code-based edited infographics ({success} successful, {fail} failed)",
    )
    for op, plan in per_op_plan.items():
        outd = plan["output_dir"]
        if os.path.isdir(outd):
            artifact.add_dir(os.path.abspath(outd), name=op)
    run.log_artifact(artifact)
    run.finish()
    print(f"W&B run uploaded: {run.url}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Code-based infographic editing: LLM modifies HTML source, then we render to PNG."
    )
    parser.add_argument("--api_keys", type=str, nargs="+", default=None,
                        help="One or more Google API Keys (randomly selected per request). "
                             "Required for online mode; ignored when --use_batch is set.")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH,
                        help="Gemini text/multimodal model used to rewrite HTML.")
    parser.add_argument("--input_file", type=str, default="editing_prompts.jsonl",
                        help="Base path of the editing-prompt file. The actual file per operation is "
                             "derived as <stem>.<operation><ext> (e.g. editing_prompts.add.jsonl).")
    parser.add_argument("--output_dir", type=str, default="edited_infographics_code",
                        help="Base output directory. Edited artifacts for each operation are written "
                             "to <output_dir>/<operation>/ (.html, .png, _thoughts.txt, .render.json).")
    parser.add_argument("--operation", type=str, default="add",
                        choices=["add", "delete", "swap_intra", "swap_inter", "text_expand", "aspect_ratio", "all"],
                        help="Which editing operation's prompts to run. 'all' iterates through every operation, "
                             "reading its per-op input file and writing to its per-op output subdir.")
    parser.add_argument("--input_mode", type=str, default="code",
                        choices=list(INPUT_MODES),
                        help="code: pass only the HTML source to the LLM. "
                             "code_image: also attach the rendered PNG so the LLM has visual context. "
                             "Output is HTML in both modes.")
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
    parser.add_argument("--num_render_workers", type=int, default=8,
                        help="Parallel Playwright workers used in the rendering phase.")
    parser.add_argument("--render_wait_ms", type=int, default=2500,
                        help="Per-page wait before screenshot (matches generate-html test script).")
    parser.add_argument("--skip_render", action="store_true",
                        help="Skip the Playwright render phase; only produce HTML files.")
    parser.add_argument("--render_only", action="store_true",
                        help="Skip LLM editing entirely; only render existing HTMLs in output_dir.")
    parser.add_argument("--force_render", action="store_true",
                        help="Re-render PNGs even if they already exist.")
    parser.add_argument("--wandb_project", type=str, default="infographic-editing-code", help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name")
    parser.add_argument("--no_wandb", action="store_true", help="Disable W&B logging")
    return parser.parse_args()


def main():
    args = parse_args()

    if getattr(args, "use_batch", False):
        raise SystemExit(
            "--use_batch is a Vertex AI feature and is not available on the "
            "openai-proxy branch. Drop --use_batch and raise --num_workers "
            "instead; requests then go to the proxy concurrently."
        )

    operations = list(OPERATIONS) if args.operation == "all" else [args.operation]

    per_op_plan = {}
    for op in operations:
        input_file = input_file_for(args.input_file, op)
        output_dir = output_dir_for(args.output_dir, op)

        if args.render_only:
            per_op_plan[op] = {"input_file": input_file, "output_dir": output_dir, "data": []}
            continue

        if not os.path.exists(input_file):
            print(f"[{op}] Input file not found: {input_file} — skipping")
            continue

        data = load_prompts(input_file, prompt_index=args.prompt_index)
        for item in data:
            item["operation"] = op

        if args.limit is not None and args.limit > 0:
            data = data[: args.limit]
            print(f"[{op}] Limited to first {args.limit} tasks")

        if not args.no_skip and os.path.isdir(output_dir):
            before = len(data)
            data = [d for d in data if not all_pngs_exist(output_dir, str(d.get("id")))]
            if before - len(data) > 0:
                print(f"[{op}] Skipping {before - len(data)} items with all {NUM_VARIANTS} variants already in {output_dir}")

        if os.path.isdir(output_dir) and data:
            no_html_ids = [
                str(d.get("id")) for d in data
                if not all_htmls_exist(output_dir, str(d.get("id")))
            ]
            html_no_png_ids = [
                str(d.get("id")) for d in data
                if all_htmls_exist(output_dir, str(d.get("id")))
                and not all_pngs_exist(output_dir, str(d.get("id")))
            ]
            print(f"[{op}] No HTML ({len(no_html_ids)} items, will call LLM): {no_html_ids}")
            print(f"[{op}] HTML but no PNG ({len(html_no_png_ids)} items, render only): {html_no_png_ids}")

        # Drop items that already have HTML — they don't need the LLM, render_all
        # picks them up directly from output_dir.
        if not args.no_skip and os.path.isdir(output_dir):
            data = [d for d in data if not all_htmls_exist(output_dir, str(d.get("id")))]

        per_op_plan[op] = {"input_file": input_file, "output_dir": output_dir, "data": data}

    total_tasks = sum(len(p["data"]) for p in per_op_plan.values())
    if total_tasks == 0 and not args.render_only and args.skip_render:
        print("Nothing to do.")
        return

    print("-" * 50)
    mode = "render-only" if args.render_only else ("batch (Vertex AI)" if args.use_batch else "online")
    print(f"Mode:        {mode}")
    print(f"Input Mode:  {args.input_mode}")
    print(f"Model:       {args.model_path}")
    print(f"Operations:  {', '.join(per_op_plan.keys())}")
    print(f"Total Tasks: {total_tasks}")
    for op, plan in per_op_plan.items():
        print(f"  - [{op}] {len(plan['data'])} tasks | in={plan['input_file']} | out={plan['output_dir']}/")
    print(f"Variants:    {NUM_VARIANTS}")
    if args.use_batch and not args.render_only:
        print(f"GCP Project:   {args.gcp_project}")
        print(f"GCP Location:  {args.gcp_location}")
        print(f"Bucket URI:    {args.batch_bucket_uri}")
    elif not args.render_only:
        print(f"Num Workers: {args.num_workers}")
        print(f"API Keys:    {len(args.api_keys) if args.api_keys else 0}")
    print(f"Render workers: {args.num_render_workers} (skip_render={args.skip_render})")
    print("-" * 50)

    if args.dry_run:
        for op, plan in per_op_plan.items():
            print(f"\n[Dry Run] operation={op} — first 3 edit prompts:\n")
            for item in plan["data"][:3]:
                print(f"--- ID {item.get('id')} | {html_path_for(item.get('image_path'))} ---")
                print(item.get("generated_edit_prompt", ""))
                print()
        return

    if not args.render_only:
        if args.use_batch:
            if not args.gcp_project or not args.batch_bucket_uri:
                print("Error: --use_batch requires --gcp_project and --batch_bucket_uri")
                return
        elif not args.api_keys:
            print("Error: --api_keys is required for online mode")
            return

    total_success = 0
    total_fail = 0
    for op, plan in per_op_plan.items():
        output_dir = plan["output_dir"]
        os.makedirs(output_dir, exist_ok=True)

        if not args.render_only and plan["data"]:
            print(f"\n=== [{op}] Editing HTML for {len(plan['data'])} items → {output_dir}/ ===")
            if args.use_batch:
                s, f = process_edits_batch(
                    project_id=args.gcp_project,
                    location=args.gcp_location,
                    bucket_uri=args.batch_bucket_uri,
                    model_path=args.model_path,
                    data_list=plan["data"],
                    output_dir=output_dir,
                    poll_interval=args.batch_poll_interval,
                    input_mode=args.input_mode,
                )
            elif args.num_workers and args.num_workers > 1:
                s, f = run_parallel(
                    args.num_workers, args.api_keys, args.model_path,
                    plan["data"], output_dir, input_mode=args.input_mode,
                )
            else:
                clients = [None]  # unused: llm_client holds the proxy client
                s, f = process_edits(
                    clients=clients,
                    model_path=args.model_path,
                    data_list=plan["data"],
                    output_dir=output_dir,
                    worker_id=0,
                    input_mode=args.input_mode,
                )
            print(f"[{op}] Edit phase: {s} success / {f} failed")
        else:
            s, f = 0, 0

        if not args.skip_render:
            print(f"\n=== [{op}] Rendering HTML to PNG ===")
            rs, rf = render_all(
                output_dir,
                num_workers=args.num_render_workers,
                wait_ms=args.render_wait_ms,
                force=args.force_render,
            )
            print(f"[{op}] Render phase: {rs} success / {rf} failed")

        total_success += s
        total_fail += f

    print(f"\nSuccess: {total_success} / {total_tasks}")
    print(f"Failed:  {total_fail} / {total_tasks}")
    print("Edited artifacts:")
    for op, plan in per_op_plan.items():
        print(f"  - [{op}] {plan['output_dir']}/")

    if not args.no_wandb and not args.render_only:
        upload_to_wandb(args, per_op_plan, total_tasks, total_success, total_fail)


if __name__ == "__main__":
    main()

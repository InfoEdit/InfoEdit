"""Recover HTML outputs from a Vertex AI Gemini batch job submitted by code/html/edit.py.

Extracts the modified HTML from each prediction's text parts and writes
``{id}_edited_1.html`` (plus ``.render.json`` with target dimensions and an
optional ``_thoughts.txt``) into --output-dir. Optionally renders the recovered
HTMLs to PNG with Playwright.

Responses are matched back to their source item by hashing the ORIGINAL_HTML
region embedded in each request's prompt (build_user_prompt always appends
``ORIGINAL_HTML:\\n{html_content}`` as the final block) and looking up the
matching item's HTML hash. Matching by the ``generated_edit_prompt`` text would
be unsafe for templated operations (e.g. aspect_ratio reuses ~3 prompts across
hundreds of items) and would silently misroute responses.

Usage:
python code/html/recover_batch.py \
    --job-name projects/.../batchPredictionJobs/... \
    --output-dir edited_infographics_code/add \
    --input-file editing_prompts.add.jsonl
"""

import argparse
import hashlib
import json
import os
import time

from edit import (
    clean_html_output,
    compute_new_dims,
    edited_paths,
    html_path_for,
    load_meta,
    load_prompts,
    render_all,
)
from google import genai


RUNNING_STATES = {"JOB_STATE_RUNNING", "JOB_STATE_PENDING", "JOB_STATE_QUEUED"}
HTML_MARKER = "ORIGINAL_HTML:\n"


def _state_name(state):
    return getattr(state, "name", None) or str(state)


def _html_hash(html_content):
    return hashlib.sha1(html_content.encode("utf-8")).hexdigest()


def _extract_original_html(prompt_text):
    """Return the ORIGINAL_HTML region embedded by build_user_prompt, or None."""
    if not prompt_text:
        return None
    idx = prompt_text.rfind(HTML_MARKER)
    if idx == -1:
        return None
    return prompt_text[idx + len(HTML_MARKER):]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-name", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--input-file", required=True,
                    help="Per-operation editing-prompt JSONL, e.g. editing_prompts.add.jsonl")
    ap.add_argument("--prompt-index", type=int, default=0)
    ap.add_argument("--project", default=os.environ.get("GCP_PROJECT"))
    ap.add_argument("--location", default="global")
    ap.add_argument("--poll-interval", type=int, default=30)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--skip-render", action="store_true",
                    help="Skip the Playwright render phase; only recover HTML files.")
    ap.add_argument("--num-render-workers", type=int, default=8)
    ap.add_argument("--render-wait-ms", type=int, default=2500)
    ap.add_argument("--force-render", action="store_true",
                    help="Re-render PNGs even if they already exist.")
    args = ap.parse_args()

    import fsspec  # requires gcsfs

    os.makedirs(args.output_dir, exist_ok=True)
    client = genai.Client(vertexai=True, project=args.project, location=args.location)
    fs = fsspec.filesystem("gcs")

    # Match each response back to its item by hashing the ORIGINAL_HTML region
    # embedded in the request prompt — see build_user_prompt() in
    # code/html/edit.py. The per-item HTML is unique, whereas
    # generated_edit_prompt is heavily templated for some operations (e.g.
    # aspect_ratio: ~3 distinct prompts across 800 items), so prompt-text
    # matching would silently map most responses onto the wrong id.
    data = load_prompts(args.input_file, prompt_index=args.prompt_index)
    html_hash_to_info = {}  # sha1(html_content) -> (item_id, new_w, new_h)
    skipped_load = 0
    for item in data:
        item_id = str(item.get("id"))
        image_path = item.get("image_path")
        html_path = html_path_for(image_path) if image_path else None
        if not html_path or not os.path.exists(html_path):
            skipped_load += 1
            continue
        try:
            with open(html_path, "r", encoding="utf-8") as fh:
                html_content = fh.read()
        except OSError:
            skipped_load += 1
            continue
        target_ar = item.get("target_aspect_ratio")
        meta = load_meta(image_path) or {}
        orig_w = int(meta.get("width", 1200))
        orig_h = int(meta.get("height", 900))
        new_w, new_h = compute_new_dims(orig_w, orig_h, target_ar) if target_ar else (orig_w, orig_h)
        h = _html_hash(html_content)
        if h in html_hash_to_info:
            # Two items with byte-identical source HTML — extremely unlikely,
            # but flag it so a silent collision can't drop responses.
            print(f"[Recover] HTML hash collision between id={item_id} and id={html_hash_to_info[h][0]}")
        html_hash_to_info[h] = (item_id, new_w, new_h)
    print(f"[Recover] Loaded {len(html_hash_to_info)} items from {args.input_file}"
          + (f" ({skipped_load} skipped — missing/unreadable HTML)" if skipped_load else ""))

    job = client.batches.get(name=args.job_name)
    while _state_name(job.state) in RUNNING_STATES:
        print(f"[Recover] state={_state_name(job.state)}; sleeping {args.poll_interval}s...")
        time.sleep(args.poll_interval)
        job = client.batches.get(name=args.job_name)

    final_state = _state_name(job.state)
    if final_state != "JOB_STATE_SUCCEEDED":
        print(f"[Recover] Job ended with state={final_state} error={job.error}")
        print(f"[Recover] Attempting to recover partial results anyway...")

    dest_uri = job.dest.gcs_uri
    print(f"[Recover] Reading predictions from {dest_uri}")
    pred_paths = fs.glob(f"{dest_uri}/*/predictions.jsonl")
    if not pred_paths:
        print(f"[Recover] No predictions.jsonl found under {dest_uri}")
        return

    success = 0
    fail = 0
    skipped = 0
    seen_ids = set()

    for pred_path in pred_paths:
        with fs.open(f"gs://{pred_path}", "r", encoding="utf-8") as f_in:
            for line in f_in:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"[Recover] Skipping malformed line: {e}")
                    fail += 1
                    continue

                prompt_text = None
                try:
                    parts = row["request"]["contents"][0]["parts"]
                    for p in parts:
                        if p.get("text"):
                            prompt_text = p["text"]
                            break
                except (KeyError, IndexError, TypeError):
                    pass

                original_html = _extract_original_html(prompt_text)
                if original_html is None:
                    print("[Recover] Skipping row: no ORIGINAL_HTML marker in prompt")
                    fail += 1
                    continue
                matched = html_hash_to_info.get(_html_hash(original_html))
                if not matched:
                    print("[Recover] Could not match row back to any input item "
                          "(ORIGINAL_HTML hash not in --input-file)")
                    fail += 1
                    continue
                item_id, new_w, new_h = matched
                seen_ids.add(item_id)

                html_save, _, txt_save, render_save = edited_paths(args.output_dir, item_id, 1)
                if os.path.exists(html_save) and not args.overwrite:
                    skipped += 1
                    continue

                status = row.get("status", "") or ""
                response = row.get("response") or {}
                edited_html = ""
                thoughts = []
                for cand in response.get("candidates", []) or []:
                    for part in (cand.get("content") or {}).get("parts", []) or []:
                        text = part.get("text")
                        if not text:
                            continue
                        if part.get("thought"):
                            thoughts.append(text)
                        else:
                            edited_html += text

                edited_html = clean_html_output(edited_html)
                if not edited_html:
                    print(f"[Recover][Failed] ID {item_id}: empty HTML in response (status={status})")
                    fail += 1
                    continue

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
                    print(f"[Recover][Failed] ID {item_id}: write error: {e}")
                    fail += 1

    missing_ids = sorted({info[0] for info in html_hash_to_info.values()
                          if info[0] not in seen_ids})
    if missing_ids:
        print(f"[Recover] {len(missing_ids)} items had no row in predictions.jsonl: "
              f"{missing_ids[:10]}{'...' if len(missing_ids) > 10 else ''}")
        fail += len(missing_ids)

    print(f"[Recover] Done. success={success} failed={fail} skipped_existing={skipped}")

    if not args.skip_render:
        print(f"\n[Recover] Rendering recovered HTML to PNG in {args.output_dir}")
        rs, rf = render_all(
            args.output_dir,
            num_workers=args.num_render_workers,
            wait_ms=args.render_wait_ms,
            force=args.force_render,
        )
        print(f"[Recover] Render phase: {rs} success / {rf} failed")


if __name__ == "__main__":
    main()

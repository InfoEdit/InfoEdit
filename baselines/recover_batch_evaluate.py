"""Recover evaluation results from a Vertex AI Gemini batch job submitted by evaluate_edits.py.

Extracts the per-(id, variant) evaluation JSON from each prediction's text parts
and appends to ``{output-file}`` in the same format produced by online evaluation.
Each response is matched back to its input item by the edited-image GCS URI
basename — evaluate_edits.py uploads each edit to
``{bucket}/batch_eval_images_{ts}/{sha10}_{id}_edited_{var}.png``, so the
trailing ``{id}_edited_{var}.png`` is unique per (item, variant). Matching by
the editing instruction text is unsafe because some operations (notably
``aspect_ratio``) share only a handful of templated prompts across hundreds of
items, which would silently map every response onto the wrong item.

Usage:
python recover_batch_evaluate.py \
    --job-name projects/.../batchPredictionJobs/... \
    --input-file editing_prompts.add.jsonl \
    --edited-dir edited_infographics/add \
    --output-file evaluation_results.add.jsonl \
    --operation add
"""

# Allow importing sibling modules from the repo root.
import sys, pathlib as _pl
sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))

import argparse
import json
import os
import re
import time

from evaluate_edits import (
    _coerce_id,
    _passthrough_metadata,
    _write_skip_record,
    load_existing_keys,
    load_prompts,
    parse_response,
    sort_output_file,
)
from google import genai


RUNNING_STATES = {"JOB_STATE_RUNNING", "JOB_STATE_PENDING", "JOB_STATE_QUEUED"}


def _state_name(state):
    return getattr(state, "name", None) or str(state)


def _extract_id_variant_from_uri(uri):
    """Pull (item_id, variant_index) from an edited-image GCS URI's basename.

    URI basenames look like ``{sha10}_{id}_edited_{var}.png`` — the
    ``\\d+_edited_\\d+`` tail is anchored to end-of-string so the hash prefix
    can't be mistaken for the id even when it happens to be all digits.
    """
    if not uri:
        return None, None
    base = uri.rsplit("/", 1)[-1]
    m = re.search(r"(\d+)_edited_(\d+)\.[A-Za-z]+$", base)
    if not m:
        return None, None
    return m.group(1), int(m.group(2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-name", required=True)
    ap.add_argument("--input-file", required=True,
                    help="Per-operation editing-prompt JSONL, e.g. editing_prompts.add.jsonl")
    ap.add_argument("--edited-dir", required=True,
                    help="Per-operation edited-images dir, e.g. edited_infographics/add")
    ap.add_argument("--output-file", required=True,
                    help="Per-operation evaluation results JSONL, e.g. evaluation_results.add.jsonl")
    ap.add_argument("--operation", required=True,
                    choices=["text_expand", "add", "swap_inter", "aspect_ratio"])
    ap.add_argument("--prompt-index", type=int, default=0)
    ap.add_argument("--project", default=os.environ.get("GCP_PROJECT"))
    ap.add_argument("--location", default="us-central1")
    ap.add_argument("--poll-interval", type=int, default=30)
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-write (id, variant) pairs already present in --output-file")
    args = ap.parse_args()

    import fsspec  # requires gcsfs

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)) or ".", exist_ok=True)
    client = genai.Client(vertexai=True, project=args.project, location=args.location)
    fs = fsspec.filesystem("gcs")

    # Match by the edited image's GCS URI basename ({sha10}_{id}_edited_{var}.png).
    # Matching by instruction text is unsafe — e.g. aspect_ratio uses only ~3
    # templated prompts across 800 items, so any prompt-based lookup would map
    # most responses onto the wrong item.
    data = load_prompts(args.input_file, prompt_index=args.prompt_index)
    id_to_item = {}
    for item in data:
        iid = item.get("id")
        if iid is None:
            continue
        item["operation"] = args.operation
        id_to_item[str(iid)] = item
    print(f"[Recover] Loaded {len(id_to_item)} items from {args.input_file}")

    done_keys = set() if args.overwrite else load_existing_keys(args.output_file)
    if done_keys:
        print(f"[Recover] {len(done_keys)} (id, variant) pairs already in "
              f"{args.output_file} will be skipped")

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
    seen = set()  # (item_id, variant)

    with open(args.output_file, "a", encoding="utf-8") as fout:
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
                        continue

                    file_uris = []
                    try:
                        parts = row["request"]["contents"][0]["parts"]
                        for p in parts:
                            fd = p.get("file_data") or p.get("fileData")
                            if fd:
                                uri = fd.get("file_uri") or fd.get("fileUri")
                                if uri:
                                    file_uris.append(uri)
                    except (KeyError, IndexError, TypeError):
                        pass

                    if len(file_uris) < 2:
                        print("[Recover] Skipping row without two file_data parts")
                        continue

                    # file_uris[0] = original, file_uris[1] = edited.
                    item_id, var_idx = _extract_id_variant_from_uri(file_uris[1])
                    if item_id is None or var_idx is None:
                        print(f"[Recover] Could not parse id/variant from edited URI: {file_uris[1]}")
                        continue

                    matched_item = id_to_item.get(item_id)
                    if not matched_item:
                        print(f"[Recover] Edited URI references id={item_id} which is not in --input-file")
                        continue

                    original_path = matched_item.get("image_path")
                    instruction = matched_item.get("generated_edit_prompt")
                    metadata = _passthrough_metadata(args.operation, matched_item)
                    edit_path = os.path.join(args.edited_dir, f"{item_id}_edited_{var_idx}.png")

                    if (item_id, var_idx) in seen:
                        continue
                    seen.add((item_id, var_idx))
                    if (item_id, -1) in done_keys or (item_id, var_idx) in done_keys:
                        skipped += 1
                        continue

                    status = row.get("status", "") or ""
                    response = row.get("response") or {}
                    text_buf = ""
                    for cand in response.get("candidates", []) or []:
                        for part in (cand.get("content") or {}).get("parts", []) or []:
                            if part.get("text"):
                                text_buf += part["text"]

                    if not text_buf.strip():
                        _write_skip_record(fout, item_id, var_idx, original_path, edit_path,
                                           instruction, f"Empty response (status={status})",
                                           metadata=metadata)
                        skipped += 1
                        continue

                    evaluation = parse_response(text_buf)
                    result = {
                        "id": _coerce_id(item_id),
                        "variant": var_idx,
                        "original_image": original_path,
                        "edited_image": edit_path,
                        "instruction": instruction,
                        **metadata,
                        "evaluation": evaluation,
                    }
                    fout.write(json.dumps(result, ensure_ascii=False) + "\n")

                    if evaluation.get("overall_judgement") == "success":
                        success += 1
                    else:
                        fail += 1

    expected_ids = set(id_to_item.keys())
    seen_ids = {iid for iid, _ in seen}
    missing_ids = sorted(expected_ids - seen_ids - {iid for iid, _ in done_keys})
    if missing_ids:
        print(f"[Recover] {len(missing_ids)} items had no row in predictions.jsonl: "
              f"{missing_ids[:10]}{'...' if len(missing_ids) > 10 else ''}")

    print(f"[Recover] Done. success={success} failed={fail} skipped={skipped}")
    sort_output_file(args.output_file)


if __name__ == "__main__":
    main()

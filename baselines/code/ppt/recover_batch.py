"""Recover code outputs from a Vertex AI Gemini batch job submitted by code/ppt/edit.py.

Extracts the python code from each prediction's text parts and writes
``{id}_edited_1.edit.py`` (and ``_thoughts.txt`` if present) into --output-dir.

Usage:
python code/ppt/recover_batch.py \
    --job-name projects/.../batchPredictionJobs/... \
    --output-dir edited_infographics_ppt/add \
    --input-file editing_prompts.add.jsonl
"""

import argparse
import json
import os
import time

from edit import clean_python_output, edited_paths, load_prompts
from google import genai


RUNNING_STATES = {"JOB_STATE_RUNNING", "JOB_STATE_PENDING", "JOB_STATE_QUEUED"}


def _state_name(state):
    return getattr(state, "name", None) or str(state)


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
    args = ap.parse_args()

    import fsspec  # requires gcsfs

    os.makedirs(args.output_dir, exist_ok=True)
    client = genai.Client(vertexai=True, project=args.project, location=args.location)
    fs = fsspec.filesystem("gcs")

    # Match by the EDITING_INSTRUCTION text — it's echoed verbatim inside each
    # response's request prompt, and is unique per item in the input file.
    data = load_prompts(args.input_file, prompt_index=args.prompt_index)
    edit_prompt_to_id = {}
    for item in data:
        ep = item.get("generated_edit_prompt")
        if ep:
            edit_prompt_to_id[ep] = str(item.get("id"))
    print(f"[Recover] Loaded {len(edit_prompt_to_id)} items from {args.input_file}")

    # Poll until terminal.
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

                item_id = None
                if prompt_text:
                    for ep, iid in edit_prompt_to_id.items():
                        if ep in prompt_text:
                            item_id = iid
                            break
                if not item_id:
                    print("[Recover] Could not match row back to any input item")
                    fail += 1
                    continue
                seen_ids.add(item_id)

                _, _, txt_save, _, code_save = edited_paths(args.output_dir, item_id, 1)
                if os.path.exists(code_save) and not args.overwrite:
                    skipped += 1
                    continue

                response = row.get("response") or {}
                edited_code = ""
                thoughts = []
                for cand in response.get("candidates", []) or []:
                    for part in (cand.get("content") or {}).get("parts", []) or []:
                        text = part.get("text")
                        if not text:
                            continue
                        if part.get("thought"):
                            thoughts.append(text)
                        else:
                            edited_code += text

                edited_code = clean_python_output(edited_code)
                if not edited_code:
                    print(f"[Recover][Failed] ID {item_id}: empty code in response")
                    fail += 1
                    continue

                with open(code_save, "w", encoding="utf-8") as fo:
                    fo.write(edited_code)
                if thoughts:
                    with open(txt_save, "w", encoding="utf-8") as fo:
                        fo.write("\n\n".join(thoughts))
                success += 1

    missing_ids = sorted(set(edit_prompt_to_id.values()) - seen_ids)
    if missing_ids:
        print(f"[Recover] {len(missing_ids)} items had no row in predictions.jsonl: "
              f"{missing_ids[:10]}{'...' if len(missing_ids) > 10 else ''}")
        fail += len(missing_ids)

    print(f"[Recover] Done. success={success} failed={fail} skipped_existing={skipped}")


if __name__ == "__main__":
    main()

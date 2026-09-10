"""Recover an OpenAI (gpt-image-*) image-edit batch whose local poller was interrupted.

pixel/gpt/edit.py submits one OpenAI Batch job per operation, then polls
inline and downloads the results. If that local process dies (Ctrl+C, SSH drop,
crash) the batch keeps running on OpenAI's side but nothing gets saved locally.
This script re-attaches to that batch by id and writes the same output files the
original run would have produced.

It reuses pixel/gpt/edit.py's exact conventions:
  - custom_id is ``item-{id}``
  - each response body carries ``data[0].b64_json``
  - decoded images are saved as ``{output_dir}/{operation}/{id}_edited_1.png``

Auth: same OPENAI_API_KEY that CREATED the batch (a batch is scoped to the
key's org/project). No interactive login needed.

Usage:
  export OPENAI_API_KEY=sk-...

  # Poll until the batch finishes, then download results (default).
  python pixel/gpt/recover_batch.py \
      --batch-id batch_6a4b48bcc3388190baccbb00ba89bd71 \
      --output_dir edited_html_infographics_gpt/v17/gpt-image-2 \
      --operation swap_inter

  # Just check status once and download if already completed (don't wait).
  python pixel/gpt/recover_batch.py --batch-id batch_... --output_dir ... \
      --operation swap_inter --no_wait
"""

import os
import json
import time
import base64
import argparse
from io import BytesIO
from datetime import datetime

from PIL import Image
from openai import OpenAI

RUNNING_STATES = {"validating", "in_progress", "finalizing", "cancelling"}


def output_dir_for(base: str, operation: str) -> str:
    """Mirror pixel/gpt/edit.py: output_dir_for: per-op subdir."""
    return os.path.join(base, operation)


def decode_output_jsonl(raw: str, output_dir: str, skip_existing: bool = True):
    """Decode an OpenAI image-edit batch output JSONL into {id}_edited_1.png.

    Mirrors process_edits_batch step 6 in pixel/gpt/edit.py so the
    recovered files are byte-for-byte compatible with a normal run.
    """
    os.makedirs(output_dir, exist_ok=True)
    success = 0
    fail = 0
    skipped = 0
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"[Recover] Skipping malformed line: {e}")
            fail += 1
            continue

        cid = rec.get("custom_id", "")
        item_id = cid[len("item-"):] if cid.startswith("item-") else cid

        if rec.get("error"):
            print(f"[Recover][Failed] ID {item_id}: {rec['error']}")
            fail += 1
            continue

        save_path = os.path.join(output_dir, f"{item_id}_edited_1.png")
        if skip_existing and os.path.exists(save_path):
            skipped += 1
            continue

        try:
            b64 = rec["response"]["body"]["data"][0]["b64_json"]
        except (KeyError, IndexError, TypeError):
            print(f"[Recover][Failed] ID {item_id}: unexpected response shape")
            fail += 1
            continue

        try:
            img = Image.open(BytesIO(base64.b64decode(b64)))
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.save(save_path)
            success += 1
        except Exception as e:
            print(f"[Recover][Failed] ID {item_id}: decode/save error: {e}")
            fail += 1

    return success, fail, skipped


def fetch_and_decode(client: OpenAI, batch_id: str, output_dir: str,
                     poll_interval: int, no_wait: bool, skip_existing: bool):
    """Download output_file_id (+ error_file_id) from the API, then decode PNGs."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch = client.batches.retrieve(batch_id)

    if not no_wait:
        while batch.status in RUNNING_STATES:
            c = batch.request_counts
            print(f"[Recover] state={batch.status}; "
                  f"completed={c.completed}/{c.total} failed={c.failed}; "
                  f"sleeping {poll_interval}s...")
            time.sleep(poll_interval)
            batch = client.batches.retrieve(batch_id)

    print(f"[Recover] FINAL status = {batch.status}")

    if batch.status in RUNNING_STATES:
        print(f"[Recover] Batch still running ({batch.status}). Re-run later, or drop "
              f"--no_wait to poll until it finishes.")
        return None

    os.makedirs(output_dir, exist_ok=True)
    if batch.error_file_id:
        err = client.files.content(batch.error_file_id).text
        err_path = os.path.join(output_dir, f"batch_errors_recover_{timestamp}.jsonl")
        with open(err_path, "w", encoding="utf-8") as f:
            f.write(err)
        print(f"[Recover] Wrote error records to {err_path}")

    if batch.status != "completed" or not batch.output_file_id:
        print(f"[Recover] Batch did not complete cleanly: status={batch.status}, "
              f"output_file_id={batch.output_file_id}")
        return None

    raw = client.files.content(batch.output_file_id).text
    raw_path = os.path.join(output_dir, f"batch_output_recover_{timestamp}.jsonl")
    with open(raw_path, "w", encoding="utf-8") as f:
        f.write(raw)
    print(f"[Recover] Saved raw output to {raw_path}")

    return decode_output_jsonl(raw, output_dir, skip_existing=skip_existing)


def main():
    parser = argparse.ArgumentParser(
        description="Recover an interrupted OpenAI gpt-image-* edit batch. "
                    "Provide EITHER --batch-id (fetch from API) OR --from-file "
                    "(decode a local output JSONL you already downloaded).")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--batch-id",
                     help="OpenAI batch id, e.g. batch_6a4b48bcc3388190baccbb00ba89bd71. "
                          "Fetches the output from the API (needs OPENAI_API_KEY).")
    src.add_argument("--from-file",
                     help="Path to a local batch output JSONL already downloaded "
                          "(no API/key needed). Each line has custom_id=item-{id} and "
                          "response.body.data[0].b64_json.")
    parser.add_argument("--output_dir", required=True,
                        help="Base output dir (WITHOUT the operation subdir), e.g. "
                             "edited_html_infographics_gpt/v17/gpt-image-2")
    parser.add_argument("--operation", required=True,
                        help="Operation name; the op subdir is appended to --output_dir "
                             "(e.g. swap_inter -> <output_dir>/swap_inter/).")
    parser.add_argument("--api_key", type=str, default=None,
                        help="OpenAI API key (defaults to $OPENAI_API_KEY). Only used with "
                             "--batch-id. Must be the same key/org that created the batch.")
    parser.add_argument("--poll_interval", type=int, default=30,
                        help="Seconds between status polls while waiting (--batch-id only).")
    parser.add_argument("--no_wait", action="store_true",
                        help="Do not poll; check status once and download only if already "
                             "completed (--batch-id only).")
    parser.add_argument("--no_skip", action="store_true",
                        help="Re-write PNGs even if the file already exists on disk.")
    args = parser.parse_args()

    output_dir = output_dir_for(args.output_dir, args.operation)
    print(f"[Recover] operation    = {args.operation}")
    print(f"[Recover] output_dir   = {output_dir}/")
    print(f"[Recover] source       = {'file:' + args.from_file if args.from_file else 'batch:' + args.batch_id}")
    print("-" * 50)

    if args.from_file:
        if not os.path.exists(args.from_file):
            raise SystemExit(f"--from-file not found: {args.from_file}")
        with open(args.from_file, "r", encoding="utf-8") as f:
            raw = f.read()
        result = decode_output_jsonl(raw, output_dir, skip_existing=not args.no_skip)
    else:
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise SystemExit("No API key. Set OPENAI_API_KEY or pass --api_key. It must be "
                             "the same key that created the batch.")
        client = OpenAI(api_key=api_key)
        result = fetch_and_decode(client, args.batch_id, output_dir,
                                  args.poll_interval, args.no_wait,
                                  skip_existing=not args.no_skip)

    if result is None:
        return
    success, fail, skipped = result
    print("-" * 50)
    print(f"[Recover] Done. saved={success} failed={fail} skipped(existing)={skipped}")
    print(f"[Recover] Images in {output_dir}/{{id}}_edited_1.png")


if __name__ == "__main__":
    main()

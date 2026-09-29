#!/usr/bin/env python
"""Aggregate per-item eval verdicts into an EC / CP / SR table.

Reads eval_results/<model>/<prefix>.<op>.detailed.jsonl (the files run_eval.sh
writes) and computes, per operation, the rates of edit_compliance (EC),
content_preservation (CP) and overall success (SR = both).

Usage:
  python evaluation/summarize_eval.py [--model MODEL] [--prefix html|ppt]
      [--ops text_expand,add,swap_inter,aspect_ratio]

Examples:
  python evaluation/summarize_eval.py --model gemini-2.5-flash-image
  python evaluation/summarize_eval.py --model gpt-5.6-sol_code --prefix ppt
"""
import json
import os
import sys

DEFAULT_MODEL = "gemini-3-pro-image-preview"
DEFAULT_OPS = ["text_expand", "add", "swap_inter", "aspect_ratio"]
DEFAULT_PREFIX = "html"   # file prefix: html_<cond>... or ppt_<cond>...


def parse_args(argv):
    model = DEFAULT_MODEL
    ops = DEFAULT_OPS
    prefix = DEFAULT_PREFIX
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--model":
            model = argv[i + 1]; i += 2
        elif a == "--ops":
            ops = argv[i + 1].split(","); i += 2
        elif a == "--prefix":
            prefix = argv[i + 1]; i += 2
        else:
            sys.exit(__doc__)
    return model, ops, prefix


def rates_for(model, op, prefix):
    """Return (n, ec, cp, sr) rates for one operation, or None if no file."""
    path = f"eval_results/{model}/{prefix}.{op}.detailed.jsonl"
    if not os.path.exists(path):
        return None
    n = ec = cp = sr = 0
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        ev = json.loads(line).get("evaluation", {})
        n += 1
        ec += ev.get("edit_compliance") == "success"
        cp += ev.get("content_preservation") == "success"
        sr += ev.get("overall_judgement") == "success"
    if n == 0:
        return None
    return n, ec / n, cp / n, sr / n


def main():
    model, ops, prefix = parse_args(sys.argv[1:])
    print(f"model={model}  source={prefix}\n")
    print(f"{'operation':<14} {'N':>4}  {'EC':>6} {'CP':>6} {'SR':>6}")
    print("-" * 42)
    srs = []
    for op in ops:
        r = rates_for(model, op, prefix)
        if r is None:
            print(f"{op:<14} {'-':>4}  {'-':>6} {'-':>6} {'-':>6}")
            continue
        n, ec, cp, sr = r
        srs.append(sr)
        print(f"{op:<14} {n:>4}  {ec*100:>5.1f}% {cp*100:>5.1f}% {sr*100:>5.1f}%")
    if srs:
        print("-" * 42)
        print(f"{'Avg SR':<14} {'':>4}  {'':>6} {'':>6} {sum(srs)/len(srs)*100:>5.1f}%")


if __name__ == "__main__":
    main()

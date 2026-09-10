#!/usr/bin/env python
"""Aggregate per-item eval verdicts into an EC / CP / SR table.

Reads eval_results/<judge_or_edit_model>/html_<condition>.<op>.detailed.jsonl and
computes, per operation, the rates of edit_compliance (EC), content_preservation
(CP) and overall success (SR = both). Pass several conditions to print them
side by side — e.g. the English baseline next to the Chinese pilot.

Usage:
  python summarize_eval.py CONDITION [CONDITION ...]
      [--model MODEL] [--prefix html|ppt] [--ops text_expand,add,swap_inter,aspect_ratio]

Examples:
  python summarize_eval.py v17 --model gemini-2.5-flash-image
  python summarize_eval.py v17_cn
  python summarize_eval.py v17 v17_cn          # baseline vs Chinese pilot
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
    conditions = []
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
            conditions.append(a); i += 1
    if not conditions:
        sys.exit(__doc__)
    return conditions, model, ops, prefix


def rates_for(model, condition, op, prefix):
    """Return (n, ec, cp, sr) rates for one condition+op, or None if no file."""
    path = f"eval_results/{model}/{prefix}_{condition}.{op}.detailed.jsonl"
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
    conditions, model, ops, prefix = parse_args(sys.argv[1:])
    print(f"model={model}\n")
    for cond in conditions:
        print(f"================ condition = {cond} ================")
        print(f"{'operation':<14} {'N':>4}  {'EC':>6} {'CP':>6} {'SR':>6}")
        print("-" * 42)
        srs = []
        for op in ops:
            r = rates_for(model, cond, op, prefix)
            if r is None:
                print(f"{op:<14} {'-':>4}  {'-':>6} {'-':>6} {'-':>6}")
                continue
            n, ec, cp, sr = r
            srs.append(sr)
            print(f"{op:<14} {n:>4}  {ec*100:>5.1f}% {cp*100:>5.1f}% {sr*100:>5.1f}%")
        if srs:
            print("-" * 42)
            print(f"{'Avg SR':<14} {'':>4}  {'':>6} {'':>6} {sum(srs)/len(srs)*100:>5.1f}%")
        print()


if __name__ == "__main__":
    main()

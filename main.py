"""
Unified experiment runner for HiEPS.

Datasets : amazon | dbpedia | wos
Methods  : flatten | path | bfs | dfs | parent | ensemble
           all_in_one | topdown_llm_beam | pointwise
           all_baseline | all_ours          (meta: runs each sub-method)

Usage:
    python main.py --dataset amazon --model gpt-4o-mini --method flatten
    python main.py --dataset amazon --model gpt-4o-mini --method all_in_one --use_desc
    python main.py --dataset amazon --model gpt-4o-mini --method topdown_llm_beam --beam_size 3
    python main.py --dataset wos    --model gpt-4o-mini --method ensemble
    python main.py --dataset wos    --model gpt-4o-mini --method all_ours --use_desc --segment 0,500
"""

import argparse
import json
import os
import subprocess
import sys
import time

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ── Constants ─────────────────────────────────────────────────────────────────

NUM_LAYERS      = {"wos": 2, "amazon": 3, "dbpedia": 3}
BASELINE        = {"flatten", "path", "bfs", "dfs", "parent", "ensemble"}
OURS            = {"all_in_one", "topdown_llm_beam", "pointwise"}
META            = {"all_baseline", "all_ours"}
ALL_METHODS     = BASELINE | OURS | META


# ── Args ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",        required=True, choices=["amazon", "dbpedia", "wos"])
    p.add_argument("--model",          required=True)
    p.add_argument("--method",         required=True, choices=sorted(ALL_METHODS))
    p.add_argument("--prompt_type",    default="single", choices=["single", "all"])
    p.add_argument("--segment",        default=None, help="e.g. 0,500")
    p.add_argument("--beam_size",      type=int, default=3)
    p.add_argument("--selection_mode", default="per_child",
                   choices=["per_child", "all_in_one", "pointwise"])
    p.add_argument("--use_desc",       action="store_true")
    p.add_argument("--max_workers",    type=int, default=10)
    p.add_argument("--yn_workers",     type=int, default=5)
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args    = parse_args()
    dataset = args.dataset
    method  = args.method
    model   = args.model

    # Meta: spawn sub-processes for each sub-method
    if method in META:
        sub_methods = (
            ["flatten", "path", "bfs", "dfs"] + ([] if dataset == "amazon" else ["parent"])
            if method == "all_baseline"
            else ["all_in_one", "topdown_llm_beam", "pointwise"]
        )
        base_argv = [a for i, a in enumerate(sys.argv[1:], 1)
                     if sys.argv[i] != "--method" and (i < 2 or sys.argv[i - 1] != "--method")]
        print(f"\n{'='*60}\n  {method}: {sub_methods}\n{'='*60}")
        for sm in sub_methods:
            print(f"\n{'─'*60}\n  >> {sm}\n{'─'*60}")
            subprocess.run([sys.executable, sys.argv[0], "--method", sm] + base_argv, check=False)
        print(f"\n{'='*60}\n  {method}: done.\n{'='*60}")
        return

    # ── Setup ─────────────────────────────────────────────────────────────────
    from config import load_dataset_config, load_label2desc, load_id2desc
    from src.utils import CallStats
    from src.method.run import run_baseline, run_ours
    from src.evaluation import evaluate_and_save

    num_layers = NUM_LAYERS[dataset]
    prefix     = dataset.upper()
    raw        = load_dataset_config(dataset)

    # Load data
    df          = pd.read_csv(f"dataset/{dataset}/eval5k.csv", header=None)
    label2id    = raw[f"{prefix}_label2id"]
    true_labels = [[label2id[v] for v in df[l].tolist()] for l in range(num_layers)]
    texts       = df[num_layers].tolist()

    i_start, i_end = (0, len(texts))
    if args.segment:
        i_start, i_end = map(int, args.segment.split(","))

    cfg = {
        "dataset":            dataset,
        "model":              model,
        "method":             method,
        "prompt_type":        args.prompt_type,
        "num_layers":         num_layers,
        "id2label":           raw[f"{prefix}_id2label"],
        "label2id":           label2id,
        "layer2ids":          raw[f"{prefix}_layer2ids"],
        "id2info":            raw[f"{prefix}_id2info"],
        "hierarchy_simplify": raw[f"{prefix}_hierarchy_simplify"],
        "id2paths":           raw.get(f"{prefix}_id2paths"),
        "id2path":            raw.get(f"{prefix}_id2path"),
        "label2desc":         load_label2desc(dataset) if args.use_desc else {},
    }
    cfg["true_labels_seg"] = [tl[i_start:i_end] for tl in true_labels]

    from src.llm_client import LLMClient
    
    stats  = CallStats(model)
    llm    = LLMClient(model=model, stats=stats)
    cfg["stats"] = stats
    cfg["llm"]   = llm

    def call_gpt(user_prompt, input_text=None, verbose=False):
        return llm.call(prompt=user_prompt, input_text=input_text, verbose=verbose)

    suffix_seg   = f"_{args.segment}" if args.segment else ""
    desc_folder  = "use_desc" if args.use_desc else "only_class_name"

    # Define Output Directory
    if method == "ensemble":
        out_dir = f"results/{dataset}/baseline/{desc_folder}/ensemble/{model}"
    elif method in BASELINE:
        out_dir = f"results/{dataset}/baseline/{desc_folder}/{method}_{args.prompt_type}/{model}"
    else:
        out_dir = (f"results/{dataset}/ours/{desc_folder}/topdown_llm_beam"
                    f"/b{args.beam_size}_{args.selection_mode}/{model}"
                    if method == "topdown_llm_beam"
                    else f"results/{dataset}/ours/{desc_folder}/{method}/{model}")

    txt_path = f"{out_dir}/eval5k{suffix_seg}.txt"
    details_path = f"{out_dir}/eval5k{suffix_seg}_details.jsonl"
    
    results = None
    # ── Check for existing results to skip LLM calls ──────────────────────────
    if os.path.exists(txt_path):
        print(f"[*] Existing results found at {txt_path}.")
        with open(txt_path, "r", errors="replace") as f:
            preds = [line.strip() for line in f]
        
        if len(preds) == (i_end - i_start):
            print(f"[*] Found {len(preds)} results. Skipping LLM calls and re-evaluating scores...")
            details = []
            if os.path.exists(details_path):
                with open(details_path, "r", encoding="utf-8") as f:
                    for line in f: details.append(json.loads(line))
            
            results = []
            for idx, p in enumerate(preds):
                d = details[idx] if idx < len(details) else {}
                results.append((i_start + idx, p, d))
        else:
            print(f"[!] Existing file length ({len(preds)}) mismatch with requested range. Re-running...")

    # ── Execute Experiment if no existing results ─────────────────────────────
    if results is None:
        if method in BASELINE:
            id2desc = load_id2desc(dataset, cfg["id2label"]) if args.use_desc else {}
            results = run_baseline(method, texts, i_start, i_end, cfg, call_gpt,
                                   args.prompt_type, args.use_desc, id2desc, args.max_workers,
                                   desc_subfolder=desc_folder, suffix_segment=suffix_seg,
                                   out_dir=out_dir)
            if method == "ensemble":
                stats.print_summary(i_end - i_start)
                return
        else:
            results = run_ours(method, texts, i_start, i_end, cfg, call_gpt, args)

    # ── Save & Evaluate ───────────────────────────────────────────────────────
    evaluate_and_save(results, cfg, out_dir, suffix_seg, is_our=(method in OURS))

    num_docs = i_end - i_start
    if stats.total_calls > 0:
        stats.print_summary(num_docs)
        stats.save(txt_path, num_docs)


if __name__ == "__main__":
    main()

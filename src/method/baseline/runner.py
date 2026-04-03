"""
Baseline runner: process_one, run_baseline, run_ensemble.

Dispatches each method to its dedicated module:
  flatten  → src/baseline/flatten.py
  path     → src/baseline/path.py
  bfs      → src/baseline/bfs.py
  dfs      → src/baseline/dfs.py
  parent   → src/baseline/parent.py
"""

import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


# ─── dispatch table ──────────────────────────────────────────────────────────

_METHOD_DISPATCH = {
    "flatten": "flatten",
    "path":    "path",
    "bfs":     "bfs",
    "dfs":     "dfs",
    "parent":  "parent",
}


# ─── single-sample helper ────────────────────────────────────────────────────

def process_one(i, text, method, dataset, prompt_type, cfg, call_gpt, use_desc, id2desc):
    """Dispatch a single text to the appropriate baseline method module."""
    if method == "flatten":
        from .flatten import process_flatten
        return process_flatten(i, text, dataset, prompt_type, cfg, call_gpt, use_desc, id2desc)
    elif method == "path":
        from .path import process_path
        return process_path(i, text, dataset, prompt_type, cfg, call_gpt, use_desc, id2desc)
    elif method == "bfs":
        from .bfs import process_bfs
        return process_bfs(i, text, dataset, prompt_type, cfg, call_gpt, use_desc, id2desc)
    elif method == "dfs":
        from .dfs import process_dfs
        return process_dfs(i, text, dataset, prompt_type, cfg, call_gpt, use_desc, id2desc)
    elif method == "parent":
        from .parent import process_parent
        return process_parent(i, text, dataset, prompt_type, cfg, call_gpt, use_desc, id2desc)
    else:
        raise ValueError(f"Unknown baseline method: {method!r}")


# ─── public runners ──────────────────────────────────────────────────────────

def run_baseline(method, texts, i_start, i_end, cfg, call_gpt,
                 prompt_type, use_desc, id2desc, max_workers):
    """Run a single baseline method in parallel. Returns sorted (idx, out, {}) list."""
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(process_one, i, texts[i], method, cfg["dataset"],
                      prompt_type, cfg, call_gpt, use_desc, id2desc): i
            for i in range(i_start, i_end)
        }
        results = []
        for fut in tqdm(as_completed(futures), total=len(futures), desc=method):
            i = futures[fut]
            try:
                idx, out = fut.result()
                results.append((idx, out, {}))
            except Exception as e:
                print(f"Error [{i}]: {e}")
                results.append((i, "ERROR", {}))
    results.sort(key=lambda x: x[0])
    return results


def run_ensemble(texts, i_start, i_end, cfg, call_gpt,
                 use_desc, id2desc, max_workers,
                 desc_subfolder, suffix_segment, out_dir):
    """
    Run ensemble: all baseline strategies → MV + PVV.
    Saves results to out_dir and returns eval_results for each tag.
    """
    from .utils import most_frequent, get_elements_and_frequencies
    from .utils import select_consistent_pathL2, select_consistent_pathL3
    from src.evaluation import evaluate_with_tools, format_results_for_display
    from src.evaluation import outputs_to_pred_labels, compute_per_class_acc

    dataset    = cfg["dataset"]
    num_layers = cfg["num_layers"]
    id2label   = cfg["id2label"]
    id2info    = cfg["id2info"]
    model      = cfg["model"]

    STRATEGIES = (
        ["flatten", "path", "bfs", "dfs"]
        if dataset == "amazon"
        else ["flatten", "path", "bfs", "dfs", "parent"]
    )

    n = i_end - i_start
    all_preds = [[[] for _ in range(n)] for _ in range(num_layers)]

    for strat in STRATEGIES:
        strat_dir  = f"results/{dataset}/baseline/{desc_subfolder}/{strat}_single/{model}"
        strat_file = f"{strat_dir}/eval5k{suffix_segment}.txt"

        if os.path.exists(strat_file):
            with open(strat_file, "r", errors="replace") as f:
                cached = [ln.rstrip("\n") for ln in f]
            if len(cached) == n:
                print(f"  >> Cached {strat}")
                preds = outputs_to_pred_labels(cached, cfg)
            else:
                print(f"  >> Cache mismatch for {strat}, re-running...")
                preds = _run_and_cache(strat, texts, i_start, i_end, cfg, call_gpt,
                                       use_desc, id2desc, max_workers, strat_dir, strat_file)
        else:
            preds = _run_and_cache(strat, texts, i_start, i_end, cfg, call_gpt,
                                   use_desc, id2desc, max_workers, strat_dir, strat_file)

        for l in range(num_layers):
            for j in range(n):
                all_preds[l][j].append(preds[l][j])

    # Majority Vote
    mv = [[] for _ in range(num_layers)]
    for l in range(num_layers):
        for j in range(n):
            tmp = [x for x in all_preds[l][j] if x != -1] or [-1]
            mv[l].append(most_frequent(tmp))

    # Path-consistent Voting
    sc_lbl  = [[] for _ in range(num_layers)]
    sc_prob = [[] for _ in range(num_layers)]
    for l in range(num_layers):
        for j in range(n):
            tmp = [x for x in all_preds[l][j] if x != -1] or [-1]
            from .utils import get_elements_and_frequencies
            el, fr = get_elements_and_frequencies(tmp)
            sc_lbl[l].append(el)
            sc_prob[l].append(fr)

    select_path = select_consistent_pathL2 if num_layers == 2 else select_consistent_pathL3
    pvv = [[] for _ in range(num_layers)]
    for j in range(n):
        if sc_lbl[0][j] == [-1]:
            for l in range(num_layers):
                pvv[l].append(-1)
            continue
        path = select_path(
            [sc_lbl[l][j]  for l in range(num_layers)],
            [sc_prob[l][j] for l in range(num_layers)],
            id2info,
        )
        for l in range(num_layers):
            pvv[l].append(path[l])

    os.makedirs(out_dir, exist_ok=True)
    base     = f"eval5k{suffix_segment}"
    all_eval = {}

    for tag, labels in [("mv", mv), ("pvv", pvv)]:
        print(f"\n{'='*50}\n  Ensemble: {tag.upper()}\n{'='*50}")

        true_seg = cfg["true_labels_seg"]
        eval_res = evaluate_with_tools(labels, true_seg, dataset)
        eval_res["per_class_accuracy"] = compute_per_class_acc(
            labels, true_seg, id2label, num_layers
        )
        print(format_results_for_display(eval_res))

        scores_path = f"{out_dir}/{base}_{tag}_scores.json"
        with open(scores_path, "w", encoding="utf-8") as f:
            json.dump(eval_res, f, indent=2, ensure_ascii=False)

        txt_path = f"{out_dir}/{base}_{tag}.txt"
        with open(txt_path, "w", errors="replace") as f:
            for j in range(n):
                row = []
                for l in range(num_layers):
                    lid = labels[l][j]
                    lbl = id2label.get(lid, "none_class")
                    if dataset == "amazon" and len(lbl) > 1 and lbl[1] == "-":
                        lbl = lbl[2:]
                    row.append(lbl)
                f.write(", ".join(row) + "\n")
        print(f"  {txt_path}\n  {scores_path}")
        all_eval[tag] = eval_res

    return all_eval


# ─── internal helper ─────────────────────────────────────────────────────────

def _run_and_cache(strat, texts, i_start, i_end, cfg, call_gpt,
                   use_desc, id2desc, max_workers, strat_dir, strat_file):
    from src.evaluation import outputs_to_pred_labels
    print(f"\n  >> Running {strat}_single...")
    results = run_baseline(strat, texts, i_start, i_end, cfg, call_gpt,
                           "single", use_desc, id2desc, max_workers)
    outputs = [o for _, o, _ in results]
    os.makedirs(strat_dir, exist_ok=True)
    with open(strat_file, "w", errors="replace") as f:
        for o in outputs:
            f.write(o + "\n")
    print(f"     Saved to {strat_file}")
    return outputs_to_pred_labels(outputs, cfg)

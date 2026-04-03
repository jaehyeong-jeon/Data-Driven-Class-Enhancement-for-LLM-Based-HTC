"""
src/method/run.py
=================
Unified entry points for running baseline and ours methods.

  run_baseline  – run a baseline strategy; ensemble is a baseline method
  run_ours      – build the right ours classifier and run it
"""

import os
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm


# ─── Baseline ────────────────────────────────────────────────────────────────

def _dispatch_baseline(i, text, method, dataset, prompt_type, cfg, call_gpt, use_desc, id2desc):
    if method == "flatten":
        from .baseline.flatten import process_flatten
        return process_flatten(i, text, dataset, prompt_type, cfg, call_gpt, use_desc, id2desc)
    elif method == "path":
        from .baseline.path import process_path
        return process_path(i, text, dataset, prompt_type, cfg, call_gpt, use_desc, id2desc)
    elif method == "bfs":
        from .baseline.bfs import process_bfs
        return process_bfs(i, text, dataset, prompt_type, cfg, call_gpt, use_desc, id2desc)
    elif method == "dfs":
        from .baseline.dfs import process_dfs
        return process_dfs(i, text, dataset, prompt_type, cfg, call_gpt, use_desc, id2desc)
    elif method == "parent":
        from .baseline.parent import process_parent
        return process_parent(i, text, dataset, prompt_type, cfg, call_gpt, use_desc, id2desc)
    else:
        raise ValueError(f"Unknown baseline method: {method!r}")


def run_baseline(method, texts, i_start, i_end, cfg, call_gpt,
                 prompt_type, use_desc, id2desc, max_workers,
                 desc_subfolder=None, suffix_segment="", out_dir=None):
    """Run a baseline method. ensemble is handled here too."""
    if method == "ensemble":
        return _run_ensemble(texts, i_start, i_end, cfg, call_gpt,
                             use_desc, id2desc, max_workers,
                             desc_subfolder, suffix_segment, out_dir)

    dataset = cfg["dataset"]
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(_dispatch_baseline, i, texts[i], method, dataset,
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


def _run_ensemble(texts, i_start, i_end, cfg, call_gpt,
                 use_desc, id2desc, max_workers,
                 desc_subfolder, suffix_segment, out_dir):
    """Run ensemble: all baseline strategies → MV + PVV."""
    from .baseline.utils import (most_frequent, get_elements_and_frequencies,
                                  select_consistent_pathL2, select_consistent_pathL3)
    from src.evaluation import (evaluate_with_tools, format_results_for_display,
                                 outputs_to_pred_labels, compute_per_class_acc)

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


# ─── Ours ─────────────────────────────────────────────────────────────────────

def run_ours(method, texts, i_start, i_end, cfg, call_gpt, args):
    """Build the appropriate ours classifier and run it."""
    from .ours.all_in_one import AllInOneClassifier
    from .ours.topdown import TopDownLLMBeamSearch
    from .ours.pointwise_classifier import PointwiseClassifier

    label2desc = cfg.get("label2desc", {})
    stats      = cfg["stats"]
    model      = cfg["model"]

    if method == "all_in_one":
        classifier = AllInOneClassifier(cfg=cfg, llm_model=model, label2desc=label2desc)
        return _run_sync(classifier, texts, i_start, i_end, args.max_workers)

    elif method == "topdown_llm_beam":
        classifier = TopDownLLMBeamSearch(
            cfg=cfg, llm_model=model, beam_size=args.beam_size,
            selection_mode=args.selection_mode,
            stats=stats, label2desc=label2desc,
        )
        return _run_async(classifier, texts, i_start, i_end, args.max_workers)

    elif method == "pointwise":
        classifier = PointwiseClassifier(
            cfg=cfg, llm_model=model,
            yn_workers=args.yn_workers,
            label2desc=label2desc, stats=stats,
        )
        return _run_async(classifier, texts, i_start, i_end, args.max_workers)

    else:
        raise ValueError(f"Unknown ours method: {method}")


def _run_sync(classifier, texts, i_start, i_end, max_workers):
    def _call(i):
        labels, details = classifier.classify(texts[i], verbose=(i == i_start))
        return (i, ", ".join(labels), details)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_call, i): i for i in range(i_start, i_end)}
        results = []
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            i = futures[fut]
            try:
                results.append(fut.result())
            except Exception as e:
                print(f"Error [{i}]: {e}")
                results.append((i, "ERROR", {}))
    results.sort(key=lambda x: x[0])
    return results


def _run_async(classifier, texts, i_start, i_end, max_workers):
    async def _gather():
        sem = asyncio.Semaphore(max_workers)

        async def _call(i):
            async with sem:
                labels, details = await classifier.classify(texts[i], verbose=(i == i_start))
            return (i, ", ".join(labels), details)

        tasks = [_call(i) for i in range(i_start, i_end)]
        return await atqdm.gather(*tasks, desc="Processing")

    raw = asyncio.run(_gather())
    results = [r if not isinstance(r, Exception) else (i_start, "ERROR", {}) for r in raw]
    results.sort(key=lambda x: x[0])
    return results

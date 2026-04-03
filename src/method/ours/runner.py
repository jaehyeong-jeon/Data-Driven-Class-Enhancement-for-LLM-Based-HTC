"""
Ours runner: run_ours builds the right classifier and returns results.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm


def run_ours(method, texts, i_start, i_end, cfg, call_gpt, args):
    """
    Build the appropriate ours classifier and run it.
    Returns sorted (idx, output_str, details) list.
    """
    from .all_in_one import AllInOneClassifier
    from .topdown import TopDownLLMBeamSearch
    from .pointwise_classifier import PointwiseClassifier

    label2desc  = cfg.get("label2desc", {})
    stats       = cfg["stats"]
    model       = cfg["model"]

    if method == "all_in_one":
        classifier = AllInOneClassifier(
            cfg=cfg, llm_model=model,
            label2desc=label2desc,
        )
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


# ─── execution helpers ────────────────────────────────────────────────────────

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

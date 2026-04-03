"""
Unified evaluation module for baseline and ours experiments.
Computes Micro-F1, Macro-F1, per-class accuracy and provides display formatting.
Contains core metrics calculation logic merged from legacy tools.py.
"""

import json
import os
import io
import numpy as np
from contextlib import redirect_stdout
from collections import defaultdict
from sklearn.metrics import precision_score, f1_score


# ── Internal Metric Primitives ─────────────────────────────────────────────

def metric_htc(y, y_pred):
    """Compute basic F1 and Precision samples."""
    zd = 0.0
    return {
        'f1_samples': f1_score(y, y_pred, average='samples', zero_division=zd),
        'f1_micro':   f1_score(y, y_pred, average='micro', zero_division=zd),
        'f1_macro':   f1_score(y, y_pred, average='macro', zero_division=zd),
        'p_samples':  precision_score(y, y_pred, average='samples', zero_division=zd)
    }


def eval_single_labels(pred_labels, true_labels, dataset):
    """
    Evaluate using flat Micro-F1 and Macro-F1.
    Prints per-layer performance to stdout.
    """
    from src.utils import read_pickle
    
    # Load dataset-native mappings
    if dataset == 'amazon':
        label2id = read_pickle('dataset/amazon/AMAZON_label2id.pkl')
        layer2id = read_pickle('dataset/amazon/AMAZON_layer2ids.pkl')
    elif dataset == 'dbpedia':
        label2id = read_pickle('dataset/dbpedia/DBPEDIA_label2id.pkl')
        layer2id = read_pickle('dataset/dbpedia/DBPEDIA_layer2ids.pkl')
    elif dataset == 'wos':
        label2id = read_pickle('dataset/wos/WOS_label2id.pkl')
        layer2id = read_pickle('dataset/wos/WOS_layer2ids.pkl')
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    layer_results = {}
    num_samples = len(pred_labels[0])
    num_total_nodes = len(label2id)

    # 1. Per-layer Evaluation
    for layer in range(len(pred_labels)):
        compare = np.array(pred_labels[layer]) == np.array(true_labels[layer])
        y_true = np.zeros((num_samples, num_total_nodes))
        y_pred = np.zeros((num_samples, num_total_nodes))
        
        for i in range(num_samples):
            y_true[i, true_labels[layer][i]] = 1
            if pred_labels[layer][i] != -1:
                y_pred[i, pred_labels[layer][i]] = 1
        
        # Calculate Macro-F1 only for the current layer's subset of IDs
        layer_mask = layer2id[layer]
        f1_macro = f1_score(y_true[:, layer_mask], y_pred[:, layer_mask], average='macro', zero_division=0.0)
        acc = float(sum(compare) / len(compare))
        
        layer_results[f'layer{layer}'] = {'macro_f1': round(f1_macro, 4), 'acc': round(acc, 4)}
        print(f'Layer {layer} Macro-F1: {f1_macro:.4f}  Acc: {sum(compare)} / {len(compare)} = {acc:.4f}')

    # 2. Overall Metrics
    y_true_all = np.zeros((num_samples, num_total_nodes))
    y_pred_all = np.zeros((num_samples, num_total_nodes))
    for i in range(num_samples):
        for layer in range(len(pred_labels)):
            y_true_all[i, true_labels[layer][i]] = 1
            if pred_labels[layer][i] != -1:
                y_pred_all[i, pred_labels[layer][i]] = 1
                
    # Special handling for Amazon (merging specific layers as per legacy tools.py logic)
    if dataset == 'amazon':
        l2_lst = []
        # Merging children layers into parent layers for legacy consistency
        for rl in ['action toy figures', 'beverages', 'cooking baking supplies', 'games', 'health care', 'herbs', 'hobbies', 'novelty gag toys', 'personal care', 'puzzles', 'skin care']:
            l1, l2 = label2id[f'1-{rl}'], label2id[f'2-{rl}']
            l2_lst.append(l2)
            y_true_all[:, l1] += y_true_all[:, l2]; y_pred_all[:, l1] += y_pred_all[:, l2]
        
        y_true_all = np.delete(y_true_all, l2_lst, axis=1) > 0
        y_pred_all = np.delete(y_pred_all, l2_lst, axis=1) > 0
    
    res = metric_htc(y_true_all.astype(float), y_pred_all.astype(float))
    print(f"Example-F1: {res['f1_samples']:.4f}\nMicro-F1: {res['f1_micro']:.4f}\nMacro-F1: {res['f1_macro']:.4f}")
    
    return {
        'layers': layer_results,
        'example_f1': round(float(res['f1_samples']), 4),
        'micro_f1': round(float(res['f1_micro']), 4),
        'macro_f1': round(float(res['f1_macro']), 4),
    }


# ── High-Level Wrappers ────────────────────────────────────────────────────

def outputs_to_pred_labels(outputs, cfg):
    """Convert list of prediction strings into per-layer label-id lists."""
    from src.utils import get_result, post_process
    
    dataset = cfg.get("dataset", "amazon")
    num_layers = cfg.get("num_layers", 2)
    method = cfg.get("method", "ours")
    prompt_type = cfg.get("prompt_type", "single")
    
    id2label = cfg["id2label"]
    label2id = cfg["label2id"]
    layer2ids = cfg["layer2ids"]
    id2info = cfg["id2info"]

    name2idx = {k.lower(): v for k, v in label2id.items()}
    strip = (lambda lbl: lbl[2:] if len(lbl) > 1 and lbl[1] in ["-", "_"] else lbl) if dataset == "amazon" else (lambda x: x)
    cands  = [[strip(id2label[nid]) for nid in layer2ids[l]] for l in range(num_layers)]

    pred_labels = [[] for _ in range(num_layers)]
    leaf_layer = num_layers - 1

    if method == "flatten":
        # 1. Predict Leaf
        for pred_str in outputs:
            if pred_str == "ERROR":
                pred_labels[leaf_layer].append(-1)
                continue
            matched = get_result(post_process(pred_str), cands[leaf_layer])
            gid = name2idx.get(f"{leaf_layer}-{matched}" if dataset == "amazon" else matched, -1)
            pred_labels[leaf_layer].append(gid)
        
        # 2. Trace Parents for upper layers
        for l in range(leaf_layer - 1, -1, -1):
            for child_id in pred_labels[l+1]:
                if child_id == -1:
                    pred_labels[l].append(-1)
                else:
                    parent = id2info.get(child_id, {}).get("parent")
                    p_id = parent[0] if isinstance(parent, list) and parent else (parent if parent is not None else -1)
                    pred_labels[l].append(p_id)
                    
    else:
        # Standard parsing for multiple layers (path, ours, etc.)
        for pred_str in outputs:
            if pred_str == "ERROR":
                for l in range(num_layers): pred_labels[l].append(-1)
                continue
            
            processed = pred_str.replace("||", ", ").split("answer:")[-1]
            parts = [p.strip() for p in processed.split(", ")]
            
            for l in range(num_layers):
                # Match against specific part if possible, otherwise whole string
                target = parts[l] if len(parts) == num_layers else processed
                matched = get_result(target, cands[l])
                gid = name2idx.get(f"{l}-{matched}" if dataset == "amazon" else matched, -1)
                pred_labels[l].append(gid)

    return pred_labels


def evaluate_with_tools(pred_labels, true_labels, dataset):
    """Capture metrics from stdout of internal eval_single_labels call."""
    f = io.StringIO()
    with redirect_stdout(f):
        eval_single_labels(pred_labels, true_labels, dataset)
    output = f.getvalue()
    
    res = {'dataset': dataset, 'num_samples': len(pred_labels[0]), 'layer_metrics': {}, 'overall_metrics': {}, 'raw_output': output}
    for line in output.strip().split('\n'):
        if line.startswith('Layer'):
            p = line.split()
            res['layer_metrics'][f"layer_{p[1]}"] = {'macro_f1': float(p[3]), 'accuracy': float(p[9]), 'correct': int(p[5]), 'total': int(p[7])}
        elif line.startswith('Example-F1:'): res['overall_metrics']['example_f1'] = float(line.split()[1])
        elif line.startswith('Micro-F1:'):   res['overall_metrics']['micro_f1'] = float(line.split()[1])
        elif line.startswith('Macro-F1:'):   res['overall_metrics']['macro_f1'] = float(line.split()[1])
    return res


def format_results_for_display(res):
    lines = ["=" * 60, "EVALUATION RESULTS", "=" * 60, f"Dataset: {res['dataset'].upper()} | Samples: {res['num_samples']}", "-" * 60]
    for lk in sorted(res['layer_metrics'].keys(), key=lambda x: int(x.split('_')[1])):
        m = res['layer_metrics'][lk]
        lines.append(f"Layer {lk.split('_')[1]}: Macro-F1={m['macro_f1']:.4f}, Acc={m['accuracy']:.4f} ({m['correct']}/{m['total']})")
    lines.append("-" * 60)
    for mk, mv in res['overall_metrics'].items(): lines.append(f"{mk.replace('_', ' ').capitalize():<12}: {mv:.4f}")
    lines.append("=" * 60)
    return "\n".join(lines)


def evaluate_and_save(results, cfg, out_dir, suffix_segment, is_our):
    dataset, num_layers, id2label, true_seg = cfg["dataset"], cfg["num_layers"], cfg["id2label"], cfg["true_labels_seg"]
    predictions = [o for _, o, _ in results]
    pred_labels = outputs_to_pred_labels(predictions, cfg)

    eval_res = evaluate_with_tools(pred_labels, true_seg, dataset)
    
    # Compute per-class accuracy
    per_class = defaultdict(lambda: {"correct": 0, "total": 0})
    for l in range(num_layers):
        for tid, pid in zip(true_seg[l], pred_labels[l]):
            if tid == -1: continue
            lbl = id2label.get(tid, str(tid))
            per_class[lbl]["total"] += 1
            if tid == pid: per_class[lbl]["correct"] += 1
    eval_res["per_class_accuracy"] = {
        lbl: {"correct": v["correct"], "total": v["total"], "accuracy": round(v["correct"] / v["total"], 4) if v["total"] else 0.0}
        for lbl, v in sorted(per_class.items(), key=lambda x: -x[1]["total"])
    }

    os.makedirs(out_dir, exist_ok=True)
    base = f"eval5k{suffix_segment}"
    with open(f"{out_dir}/{base}.txt", "w", errors="replace") as f:
        for _, o, _ in results: f.write(o + "\n")
    if is_our:
        with open(f"{out_dir}/{base}_details.jsonl", "w", encoding="utf-8") as f:
            for _, _, d in results: f.write(json.dumps(d, ensure_ascii=False, default=str) + "\n")
    with open(f"{out_dir}/{base}_scores.json", "w", encoding="utf-8") as f:
        json.dump(eval_res, f, indent=2, ensure_ascii=False)

    print("\n" + format_results_for_display(eval_res))
    if eval_res["per_class_accuracy"]:
        print("\n-- Per-class Accuracy ──────────────────────────")
        for lbl, v in list(eval_res["per_class_accuracy"].items())[:20]: # Show top 20
            bar = "█" * int(v["accuracy"] * 20)
            print(f"  {lbl:<30} {v['correct']:>4}/{v['total']:<4}  {v['accuracy']:.0%} {bar}")
    print(f"\nSaved results to {out_dir}")
    return eval_res

"""
Baseline-specific utilities for ensemble voting and prompt building.
"""

import math
import functools
import numpy as np
from collections import Counter


def most_frequent(lst):
    if not lst: return -1
    return Counter(lst).most_common(1)[0][0]


def get_elements_and_frequencies(lst):
    counter = Counter(lst)
    elements = list(counter.keys())
    frequencies = list(np.array(list(counter.values())) / len(lst))
    return elements, frequencies


def list_lcm(numbers):
    return functools.reduce(math.lcm, numbers)


def select_consistent_pathL2(pred_labels, pred_probs, id2info):
    """Path-consistent voting for 2-layer hierarchies."""
    paths, scores = [], []
    for l0, s0 in zip(pred_labels[0], pred_probs[0]):
        for l1, s1 in zip(pred_labels[1], pred_probs[1]):
            if l1 in id2info[l0]['children']:
                paths.append([l0, l1])
                scores.append(s0 * s1 * s1)
            elif l1 == -1:
                paths.append([l0, -1])
                scores.append(s0 * 0.0001)
    return paths[np.argmax(scores)] if scores else [-1, -1]


def select_consistent_pathL3(pred_labels, pred_probs, id2info):
    """Path-consistent voting for 3-layer hierarchies (Amazon/DBPedia)."""
    paths, scores = [], []
    for l0, s0 in zip(pred_labels[0], pred_probs[0]):
        for l1, s1 in zip(pred_labels[1], pred_probs[1]):
            if l1 in id2info[l0]['children']:
                for l2, s2 in zip(pred_labels[2], pred_probs[2]):
                    if l2 in id2info[l1]['children']:
                        paths.append([l0, l1, l2])
                        scores.append(s0 * s1**2 * s2**3)
                    elif l2 == -1:
                        paths.append([l0, l1, -1])
                        scores.append(s0 * s1**2 * 1e-6)
            elif l1 == -1:
                paths.append([l0, -1, -1])
                scores.append(s0 * 1e-10)
    return paths[np.argmax(scores)] if scores else [-1, -1, -1]


def build_baseline_desc_block(candidate_ids, id2desc, id2label, dataset):
    """Build category description block for baseline prompts."""
    from src.utils import get_node_label
    lines = []
    for cid in candidate_ids:
        desc = id2desc.get(cid, "")
        if desc:
            lbl = get_node_label(cid, id2label, dataset)
            lines.append(f"- {lbl}: {desc}")
    return "\n\nCategory descriptions:\n" + "\n".join(lines) if lines else ""

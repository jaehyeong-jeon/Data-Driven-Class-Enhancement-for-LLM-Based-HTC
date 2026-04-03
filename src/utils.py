"""
General utility functions for taxonomy navigation, response parsing, data I/O, and ensemble voting.
Merged from legacy tools.py and ours/utils.py.
"""

import json
import pickle
import math
import functools
import threading
import numpy as np
from collections import Counter

# ── Taxonomy & Formatting ──────────────────────────────────────────────────

def get_node_label(node_id, id2label, dataset='amazon'):
    """Get cleaned label for a node (removes prefixes/suffixes according to dataset)."""
    label = id2label[node_id]
    if dataset == 'amazon':
        if len(label) > 1 and label[1] == '-': return label[2:]
        if len(label) > 1 and label[1] == '_': return label[2:]
        if len(label) > 2 and label[2] == '_': return label[3:]
        if len(label) > 3 and label[3] == '_': return label[4:]
    return label


def get_node_path(node_id, id2info):
    """Get the full path from root to the given node_id."""
    path = []
    current = node_id
    while current is not None:
        path.insert(0, current)
        parent = id2info[current].get('parent')
        if parent is None or parent == -1: break
        if isinstance(parent, list):
            if len(parent) == 0: break
            parent = parent[0]
        current = parent
    return path


def first_letter_lowercase(s):
    return s[0].lower() + s[1:] if s else s


def post_process(output):
    """Clean LLM output string."""
    if not output: return ''
    output = output.strip().replace('\n', ' ')
    if output.endswith(('.', ',')): output = output[:-1]
    return output.strip().lower()


# ── Response Parsing ───────────────────────────────────────────────────────

def get_result(text: str, class_names: list):
    """Fuzzy match class_names in text based on occurrence counts."""
    t_lst = []
    for cn in class_names:
        t_lst.append(text.lower().count(cn.lower()))
    
    if sum(t_lst) == 0:
        return 'none_class'
    
    if len(t_lst) == 1:
        return class_names[0].lower()
        
    indices = np.argsort(t_lst)[-2:]
    idx0, idx1 = indices[0], indices[1]
    if t_lst[idx0] == t_lst[idx1] and (class_names[idx0] in class_names[idx1] or class_names[idx1] in class_names[idx0]):
        res = class_names[idx0] if len(class_names[idx0]) > len(class_names[idx1]) else class_names[idx1]
        return res.lower()
    return class_names[np.argmax(t_lst)].lower()


def parse_selection_response(response, candidate_names):
    """Parse a simple selection response into a candidate index."""
    resp = post_process(response)
    matched = get_result(resp, candidate_names)
    if matched == 'none_class': return 0
    for idx, name in enumerate(candidate_names):
        if name.lower() == matched: return idx
    return 0


def parse_ranking_response(response, candidate_names):
    """Parse ranking response into sorted indices."""
    parts = [part.strip() for part in response.split('>')]
    indices, seen = [], set()
    for part in parts:
        matched_name = get_result(post_process(part), candidate_names)
        if matched_name == 'none_class': continue
        for idx, name in enumerate(candidate_names):
            if name.lower() == matched_name and idx not in seen:
                indices.append(idx); seen.add(idx); break
    missing = [i for i in range(len(candidate_names)) if i not in seen]
    indices.extend(missing)
    return indices


# ── Data I/O ───────────────────────────────────────────────────────────────

def read_json(path):
    with open(path, 'r', encoding='utf-8') as f: return json.load(f)

def write_json(data, path):
    with open(path, 'w', encoding='utf-8') as f: json.dump(data, f, ensure_ascii=False)

def read_pickle(filename):
    with open(filename, 'rb') as f: return pickle.load(f)

def save_pickle(data, filename):
    with open(filename, 'wb') as f: pickle.dump(data, f)


# ── Generic LLM Call Statistics Tracker ───────────────────────────────────

# USD per 1M tokens: {model_id: (input, output)}
_PRICING = {
    "gpt-3.5-turbo":           (0.50,   1.50),
    "gpt-4o-mini":             (0.15,   0.60),
    
    # GPT-5 Series
    "gpt-5":                   (1.25,  10.00),
    "gpt-5.1":                 (1.25,  10.00),
    "gpt-5.2":                 (1.75,  14.00),
    "gpt-5.2-pro":             (21.00, 168.0),
    "gpt-5-mini":              (0.25,   2.00),
    "gpt-5-nano":              (0.05,   0.40),
    "gpt-5-pro":               (15.00, 120.0),
    
    # GPT-5.4 Series
    "gpt-5.4":                 (2.50,  15.00),
    "gpt-5.4-mini":            (0.75,   4.50),
    "gpt-5.4-nano":            (0.20,   1.25),
    "gpt-5.4-pro":             (30.00, 180.0),
}


class CallStats:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._lock = threading.Lock()
        self.total_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def record(self, usage):
        """Record stats from one successful LLM call.

        Args:
            usage: completion.usage object (has .prompt_tokens / .completion_tokens),
                   or None if unavailable.
        """
        with self._lock:
            self.total_calls += 1
            if usage is not None:
                self.total_input_tokens  += getattr(usage, 'prompt_tokens',     0)
                self.total_output_tokens += getattr(usage, 'completion_tokens', 0)

    # ── Derived metrics ───────────────────────────────────────────────────────

    def _cost(self):
        prices = _PRICING.get(self.model_name)
        if prices is None:
            return None
        return (self.total_input_tokens  / 1000000 * prices[0] +
                self.total_output_tokens / 1000000 * prices[1])

    def summary(self, num_docs: int) -> dict:
        cost = self._cost()
        return {
            "model":                       self.model_name,
            "num_documents":               num_docs,
            "total_llm_calls":             self.total_calls,
            "avg_calls_per_doc":           round(self.total_calls / num_docs, 3) if num_docs else 0,
            "total_input_tokens":          self.total_input_tokens,
            "total_output_tokens":         self.total_output_tokens,
            "avg_input_tokens_per_call":   round(self.total_input_tokens  / self.total_calls, 1) if self.total_calls else 0,
            "avg_output_tokens_per_call":  round(self.total_output_tokens / self.total_calls, 1) if self.total_calls else 0,
            "estimated_cost_usd":          round(cost, 4)           if cost is not None else "unknown (model not in pricing table)",
            "cost_per_doc_usd":            round(cost / num_docs, 6) if (cost and num_docs) else "unknown",
        }

    def save(self, output_file: str, num_docs: int):
        """Save stats as JSON next to output_file (replaces .txt/.json → _stats.json)."""
        import os
        base = os.path.splitext(output_file)[0]
        stats_file = base + "_call_stats.json"
        data = self.summary(num_docs)
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        return stats_file

    def print_summary(self, num_docs: int):
        d = self.summary(num_docs)
        print(f"\n── LLM Call Statistics ──────────────────")
        print(f"  Documents processed : {d['num_documents']}")
        print(f"  Total LLM calls     : {d['total_llm_calls']}")
        print(f"  Avg calls / doc     : {d['avg_calls_per_doc']}")
        print(f"  Total input tokens  : {d['total_input_tokens']}")
        print(f"  Total output tokens : {d['total_output_tokens']}")
        print(f"  Avg input  / call   : {d['avg_input_tokens_per_call']}")
        print(f"  Avg output / call   : {d['avg_output_tokens_per_call']}")
        print(f"  Estimated cost      : ${d['estimated_cost_usd']}")
        print(f"  Cost per document   : ${d['cost_per_doc_usd']}")
        print(f"─────────────────────────────────────────")

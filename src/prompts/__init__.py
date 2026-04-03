import json
import os
from jinja2 import Environment, FileSystemLoader

_TEMPLATE_DIR = os.path.dirname(os.path.abspath(__file__))

_env = Environment(
    loader=FileSystemLoader(_TEMPLATE_DIR),
    trim_blocks=True,
    lstrip_blocks=True,
    autoescape=False,
)

# Inline snippet constants (used as .format() strings in baseline parent method)
FINE_TEMPLATE = "Each of these candidate categories contains a set of fine-grained sub-categories as follows: {}. "
COARSE_TEMPLATE = "These candidate categories belong to the same coarse-grained category: {}. "


def _render(template_path, **kwargs):
    return _env.get_template(template_path).render(**kwargs)


def _generate_bfs_content(dataset, id2label, layer2ids, id2info):
    """Generate content blocks for the BFS template."""
    if dataset == 'wos':
        l1 = f"[{', '.join([id2label[idx] for idx in layer2ids[0]])}]"
        l2 = ""
        for idx in layer2ids[0]:
            children = ', '.join([id2label[cid] for cid in id2info[idx]['children']])
            l2 += f"\n  [{children}] (subdomains of {id2label[idx]}); "
        return l1, l2
    else:
        def lbl(nid):
            return id2label[nid][2:] if dataset == 'amazon' else id2label[nid]

        l1 = f"[{', '.join([lbl(idx) for idx in layer2ids[0]])}]"
        l2 = ""
        for idx in layer2ids[0]:
            children = ', '.join([lbl(cid) for cid in id2info[idx]['children']])
            l2 += f"\n  [{children}] (subcategories of {lbl(idx)}); "
        l3 = ""
        for idx in layer2ids[1]:
            children = ', '.join([lbl(cid) for cid in id2info[idx]['children']])
            l3 += f"\n  [{children}] (subcategories of {lbl(idx)}); "
        return l1, l2, l3


class PromptManager:
    def __init__(self, dataset):
        self.dataset = dataset

    def _r(self, path, **kwargs):
        return _render(path, dataset=self.dataset, **kwargs)

    # ── Baseline ──────────────────────────────────────────────────────────────

    def flatten(self, variant, candidates):
        """variant: 'single'|'add'|'all', candidates: pre-joined string"""
        return self._r(f'baseline/flatten/{variant}.j2', candidates=candidates)

    def path_prompt(self, variant, candidates):
        """variant: 'single'|'add'|'all', candidates: pre-joined string"""
        return self._r(f'baseline/path/{variant}.j2', candidates=candidates)

    def bfs(self, id2label, layer2ids, id2info):
        args = _generate_bfs_content(self.dataset, id2label, layer2ids, id2info)
        if self.dataset == 'wos':
            l1, l2 = args
            return self._r('baseline/bfs.j2', l1=l1, l2=l2, l3='')
        else:
            l1, l2, l3 = args
            return self._r('baseline/bfs.j2', l1=l1, l2=l2, l3=l3)

    def dfs(self, hierarchy_simplify):
        hierarchy_json = json.dumps(hierarchy_simplify, indent=2)
        return self._r('baseline/dfs.j2', hierarchy_json=hierarchy_json)

    # ── Ours ──────────────────────────────────────────────────────────────────

    def ranker(self, text, candidates):
        """Returns (system_prompt, user_prompt) for RankGPT-style ranking."""
        system = self._r('ours/ranker/system.j2')
        user = self._r('ours/ranker/user.j2', text=text, candidates=candidates)
        return system, user

    def pointwise_yn(self, text, path_label):
        return self._r('ours/pointwise/yn.j2', text=text, path_label=path_label)

    def final(self, text, candidates):
        """Final single-selection from filtered candidates (pointwise and topdown)."""
        return self._r('ours/final.j2', text=text, candidates=candidates)


# ── Module-level functions (used by ours/ classifiers) ────────────────────────

def create_ranker_prompts(text, candidates, dataset='amazon'):
    return PromptManager(dataset).ranker(text, candidates)


def create_pointwise_yn_prompt(text, path_label, dataset='amazon'):
    return PromptManager(dataset).pointwise_yn(text, path_label)


def create_pointwise_final_prompt(text, candidates, dataset='amazon'):
    return PromptManager(dataset).final(text, candidates)

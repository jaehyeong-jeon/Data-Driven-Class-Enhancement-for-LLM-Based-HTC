"""
Parent Baseline Method
======================
Layer-by-layer classification: at each layer, the LLM is given the children of
the previously predicted parent node and asked to pick one.

Number of LLM calls per sample: num_layers.

Supports both prompt_type='single' and prompt_type='all' (multi-label per layer).
Handles Amazon (3-layer, multi-parent), DBpedia (3-layer), and WOS (2-layer).
"""


def process_parent(i, text, dataset, prompt_type, cfg, call_gpt, use_desc, id2desc):
    """
    Parent method: layer-by-layer LLM classification conditioned on parent predictions.

    Args:
        i: sample index
        text: raw text
        dataset: 'amazon' | 'dbpedia' | 'wos'
        prompt_type: 'single' | 'all'
        cfg: config dict
        call_gpt: callable
        use_desc: bool
        id2desc: {node_id: desc_str}

    Returns:
        (i, output_str)  where output_str = ", ".join(parent_labels)
    """
    from src.utils import post_process, get_result, first_letter_lowercase
    from .utils import build_baseline_desc_block
    from src.prompts import PromptManager, FINE_TEMPLATE, COARSE_TEMPLATE

    num_layers  = cfg["num_layers"]
    id2label    = cfg["id2label"]
    label2id    = cfg["label2id"]
    layer2ids   = cfg["layer2ids"]
    id2info     = cfg["id2info"]
    pm          = PromptManager(dataset)
    name2idx    = {k.lower(): v for k, v in label2id.items()}
    sentence    = text.replace("\n", " ").replace('"', '\\"')

    def _lbl(nid):
        lbl = id2label[nid]
        return lbl[2:] if dataset == "amazon" else lbl

    parent_ids, parent_labels = [], []

    for layer in range(num_layers):
        if layer == 0:
            _cids = layer2ids[0]
            cands = [_lbl(n) for n in _cids]
        else:
            if prompt_type == "single":
                par = parent_ids[layer - 1]
                if par == -1:
                    parent_ids.append(-1)
                    parent_labels.append("none_class")
                    continue
                _cids = id2info[par]["children"]
                cands = [_lbl(n) for n in _cids]
            elif prompt_type == "all" and dataset == "amazon":
                if layer == 1:
                    par = parent_ids[0]
                    if par == [-1]:
                        parent_ids.append([-1])
                        parent_labels.append("none_class")
                        continue
                    _cids = [c for p in par for c in id2info[p]["children"]]
                    cands = [f"{_lbl(p)}, {_lbl(c)}" for p in par for c in id2info[p]["children"]]
                else:  # layer == 2
                    p0, p1 = parent_ids[0], parent_ids[1]
                    if p1 == [-1]:
                        parent_ids.append([-1])
                        parent_labels.append("none_class")
                        continue
                    anc   = [[a, b] for a in p0 for b in p1 if b in id2info[a]["children"]]
                    _cids = [c for a in anc for c in id2info[a[1]]["children"]]
                    cands = [f"{_lbl(a[0])}, {_lbl(a[1])}, {_lbl(c)}" for a in anc for c in id2info[a[1]]["children"]]
            else:  # all, non-amazon
                par = parent_ids[layer - 1]
                if par == [-1]:
                    parent_ids.append([-1])
                    parent_labels.append("none_class")
                    continue
                _cids = [c for p in par for c in id2info[p]["children"]]
                cands = (
                    [f"{_lbl(p)}, {_lbl(c)}" for p in par for c in id2info[p]["children"]]
                    if dataset == "wos" and layer == 1
                    else [_lbl(c) for c in _cids]
                )

        # Build prompt
        if (prompt_type == "all" and dataset == "amazon" and layer in [1, 2]) or \
           (prompt_type == "all" and dataset == "wos" and layer == 1):
            user_prompt = pm.path_prompt(prompt_type, "; ".join(cands))
        else:
            user_prompt = pm.flatten(prompt_type, ", ".join(cands))
            if layer != 0 and prompt_type == "single":
                user_prompt += COARSE_TEMPLATE.format(
                    ": ".join(_lbl(a) if a != -1 else "none" for a in parent_ids)
                )
            if layer != num_layers - 1:
                fine_info = []
                for c in cands:
                    cid = label2id.get(f"{layer}-{c}" if dataset == "amazon" else c)
                    if cid is not None:
                        fine_info.append(
                            f"{c}: ({', '.join(_lbl(ch) for ch in id2info[cid]['children'])})"
                        )
                fine_prompt = FINE_TEMPLATE.format("; ".join(fine_info))
                if layer == 0 or prompt_type == "all":
                    user_prompt += fine_prompt
                else:
                    user_prompt += " Moreover, " + first_letter_lowercase(fine_prompt)

        if use_desc:
            user_prompt += build_baseline_desc_block(_cids, id2desc, id2label, dataset)

        raw = post_process(call_gpt(user_prompt, sentence, verbose=(i == 0 and layer == 0)))

        if prompt_type == "single":
            gl  = get_result(raw, cands)
            gid = name2idx.get(f"{layer}-{gl}" if dataset == "amazon" else gl, -1)
            parent_ids.append(gid)
            parent_labels.append(gl)
        else:
            gls  = [get_result(o, cands) for o in raw.split("||")]
            gids = (
                [name2idx.get(f"{layer}-{g.split(', ')[-1]}", -1) for g in gls]
                if dataset == "amazon"
                else [name2idx.get(g.split(", ")[-1], -1) for g in gls]
            )
            parent_ids.append(gids)
            parent_labels.append(
                "none_class" if gids == [-1]
                else "||".join(_lbl(g) for g in gids)
            )

    output = ", ".join(parent_labels)
    return (i, output)

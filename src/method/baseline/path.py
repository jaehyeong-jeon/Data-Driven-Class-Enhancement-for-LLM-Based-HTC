"""
Path Baseline Method
====================
Single LLM call: all root→leaf taxonomy paths presented as a semicolon-separated list.
The LLM picks one complete path.

One LLM call per sample.
"""


def process_path(i, text, dataset, prompt_type, cfg, call_gpt, use_desc, id2desc):
    """
    Path method: present all root→leaf paths and ask the LLM to pick one.

    Args:
        i: sample index
        text: raw text
        dataset: 'amazon' | 'dbpedia' | 'wos'
        prompt_type: 'single' | 'all'
        cfg: config dict (id2label, layer2ids, id2paths, id2path, ...)
        call_gpt: callable
        use_desc: bool
        id2desc: {node_id: desc_str}

    Returns:
        (i, output_str)
    """
    from src.utils import post_process
    from .utils import build_baseline_desc_block
    from src.prompts import PromptManager

    num_layers  = cfg["num_layers"]
    id2label    = cfg["id2label"]
    layer2ids   = cfg["layer2ids"]
    id2paths    = cfg.get("id2paths")
    id2path     = cfg.get("id2path")
    pm          = PromptManager(dataset)
    sentence    = text.replace("\n", " ").replace('"', '\\"')

    leaf_ids = layer2ids[num_layers - 1]

    if id2paths:
        cands = [p for n in leaf_ids for p in id2paths.get(n, [])]
    else:
        cands = [", ".join(id2path[n]) for n in leaf_ids]

    user_prompt = pm.path_prompt(prompt_type, "; ".join(cands))

    if use_desc:
        user_prompt += build_baseline_desc_block(leaf_ids, id2desc, id2label, dataset)

    output = post_process(call_gpt(user_prompt, sentence, verbose=(i == 0)))
    return (i, output)

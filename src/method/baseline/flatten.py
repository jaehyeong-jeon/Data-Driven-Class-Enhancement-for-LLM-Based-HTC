"""
Flatten Baseline Method
=======================
Single LLM call: all leaf labels presented as a flat comma-separated list.
The LLM picks one.

One LLM call per sample.
"""


def process_flatten(i, text, dataset, prompt_type, cfg, call_gpt, use_desc, id2desc):
    """
    Flatten method: present all leaf nodes as a flat list and ask the LLM to pick one.

    Args:
        i: sample index
        text: raw text
        dataset: 'amazon' | 'dbpedia' | 'wos'
        prompt_type: 'single' | 'all'
        cfg: config dict (id2label, layer2ids, ...)
        call_gpt: callable (user_prompt, input_text) -> str
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
    pm          = PromptManager(dataset)
    sentence    = text.replace("\n", " ").replace('"', '\\"')

    def _lbl(nid):
        lbl = id2label[nid]
        return lbl[2:] if dataset == "amazon" else lbl

    leaf_ids    = layer2ids[num_layers - 1]
    user_prompt = pm.flatten(prompt_type, ", ".join(_lbl(n) for n in leaf_ids))

    if use_desc:
        user_prompt += build_baseline_desc_block(leaf_ids, id2desc, id2label, dataset)

    output = post_process(call_gpt(user_prompt, sentence, verbose=(i == 0)))
    return (i, output)

"""
DFS Baseline Method
===================
Single LLM call: hierarchy presented in DFS (depth-first / simplified) order.
The LLM picks one leaf.

One LLM call per sample.
"""


def process_dfs(i, text, dataset, prompt_type, cfg, call_gpt, use_desc, id2desc):
    """
    DFS method: present the simplified hierarchy in depth-first order.

    Args:
        i: sample index
        text: raw text
        dataset: 'amazon' | 'dbpedia' | 'wos'
        prompt_type: 'single' | 'all'
        cfg: config dict (hierarchy_simplify, ...)
        call_gpt: callable
        use_desc: bool
        id2desc: {node_id: desc_str}

    Returns:
        (i, output_str)
    """
    from src.utils import post_process
    from .utils import build_baseline_desc_block
    from src.prompts import PromptManager

    num_layers   = cfg["num_layers"]
    id2label     = cfg["id2label"]
    layer2ids    = cfg["layer2ids"]
    hier_simple  = cfg["hierarchy_simplify"]
    pm           = PromptManager(dataset)
    sentence     = text.replace("\n", " ").replace('"', '\\"')

    leaf_ids    = layer2ids[num_layers - 1]
    user_prompt = pm.dfs(hier_simple)

    if use_desc:
        user_prompt += build_baseline_desc_block(leaf_ids, id2desc, id2label, dataset)

    output = post_process(call_gpt(user_prompt, sentence, verbose=(i == 0)))
    return (i, output)

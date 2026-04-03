"""
Pointwise Classifier
====================
For each leaf node in the taxonomy, independently ask the LLM whether
the document is relevant (Yes/No).  Then collect all "Yes" leaves and
make one final LLM call to pick the single best one.

Handles Amazon's multi-parent nodes via id2paths: each possible path
for a multi-parent leaf becomes a separate yes/no candidate.

Number of LLM calls per document: len(expanded_candidates) + 1
"""

import asyncio
import re

from src.utils import get_node_label, get_node_path


class PointwiseClassifier:
    """
    Pointwise relevance filtering + final LLM selection (async).

    Step 1 (parallel): For every (leaf, path) pair ask the LLM "Yes or No?"
    Step 2 (single):   Among all Yes entries, ask the LLM to pick the best one.
    Fallback: If zero entries are Yes, the final call uses ALL entries.
    """

    def __init__(
        self,
        cfg,
        llm_client,
        llm_model,
        yn_workers=100,
        stats=None,
        label2desc=None,
    ):
        self.llm_model  = llm_model
        self.id2label   = cfg["id2label"]
        self.layer2ids  = cfg["layer2ids"]
        self.id2info    = cfg["id2info"]
        self.dataset    = cfg["dataset"]
        self.num_layers = cfg["num_layers"]
        self.id2paths   = cfg.get("id2paths")
        self.yn_workers = yn_workers
        self.stats      = stats
        self.label2desc = label2desc or {}
        self.llm        = cfg["llm"]

        self._candidates = self._build_candidates()

    # ── Candidate building ────────────────────────────────────────────────────

    def _build_candidates(self):
        """Build (leaf_id, path_label_str) pairs, expanding multi-parent nodes."""
        leaf_ids = self.layer2ids[self.num_layers - 1]
        candidates = []
        for lid in leaf_ids:
            if self.id2paths is not None:
                paths = self.id2paths.get(lid, [])
                if paths:
                    for p in paths:
                        candidates.append((lid, p.replace(", ", " > ")))
                    continue
            path = get_node_path(lid, self.id2info)
            labels = [get_node_label(nid, self.id2label, self.dataset) for nid in path]
            candidates.append((lid, " > ".join(labels)))
        return candidates

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _labels_from_path_str(self, path_str):
        """Convert a path string like 'beauty > skin care > face' to per-layer labels."""
        from src.utils import get_result
        parts = [p.strip() for p in path_str.split(" > ")]

        candidates_per_layer = [
            [get_node_label(nid, self.id2label, self.dataset)
             for nid in self.layer2ids[l]]
            for l in range(self.num_layers)
        ]

        labels = []
        for layer in range(self.num_layers):
            if layer < len(parts):
                matched = get_result(parts[layer], candidates_per_layer[layer])
                labels.append(matched if matched != 'none_class' else parts[layer].lower())
            else:
                labels.append('none_class')
        return labels

    # ── Async LLM calls ──────────────────────────────────────────────────────

    async def _ask_yn(self, text, path_label):
        """Ask a single yes/no question for one path. Returns (path_label, bool)."""
        from .prompts import create_pointwise_yn_prompt
        prompt = create_pointwise_yn_prompt(text, path_label, self.dataset)
        
        if self.label2desc:
            leaf = path_label.split(" > ")[-1].strip().lower()
            desc = self.label2desc.get(leaf, "")
            if desc:
                prompt = prompt.replace(
                    "Your answer:",
                    f"\nCategory description for '{leaf}':\n  {desc}\n\nYour answer:"
                )


        response = await self.llm.acall(prompt=prompt, max_tokens=50, verbose=False)
        return path_label, "yes" in response.lower()

    async def _final_select(self, text, candidates, verbose=False):
        """Pick one from candidates list of (leaf_id, path_label). Returns index."""
        from .prompts import create_pointwise_final_prompt
        candidates_info = [
            {'id': lid, 'path_label': plabel}
            for lid, plabel in candidates
        ]
        prompt = create_pointwise_final_prompt(text, candidates_info, self.dataset)

        if self.label2desc:
            desc_block = "\n\nDescriptions for each candidate path:\n"
            for i, (_, plabel) in enumerate(candidates, 1):
                desc_block += f"\n[{i}. {plabel}]\n"
                for part in plabel.split(" > "):
                    desc = self.label2desc.get(part.strip().lower(), "")
                    if desc:
                        desc_block += f"  ({part.strip()}): {desc}\n"
            desc_block += "\nSelect the SINGLE MOST APPROPRIATE category. "
            desc_block += "Respond with ONLY the number. Do NOT output descriptions or keywords."
            prompt = prompt.replace("Your answer:", desc_block + "\n\nYour answer:")

        if verbose:
            print(f"\n=== Pointwise Final Prompt ===\n{prompt}\n")

        response = await self.llm.acall(prompt=prompt, max_tokens=200, verbose=verbose)
        if verbose:
            print(f"\n=== Pointwise Final Response ===\n{response}\n")
        return self._parse_index(response, candidates)

    def _parse_index(self, response, candidates):
        patterns = [
            r'(?:Answer|answer):\s*(\d+)',
            r'(?:Selected|selected):\s*(\d+)',
            r'^\s*(\d+)\s*$',
            r'(\d+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                idx = int(match.group(1))
                if 1 <= idx <= len(candidates):
                    return idx - 1
        return 0

    # ── Main classify ─────────────────────────────────────────────────────────

    async def classify(self, text, verbose=False):
        original_text = text.replace("\n", " ").replace('"', '\\"')

        yn_sem = asyncio.Semaphore(self.yn_workers)

        async def _ask_yn_sem(path_label):
            async with yn_sem:
                return await self._ask_yn(original_text, path_label)

        yn_tasks = [_ask_yn_sem(plabel) for _lid, plabel in self._candidates]
        yn_results = await asyncio.gather(*yn_tasks)

        yes_entries = []
        for (lid, plabel), (_, is_yes) in zip(self._candidates, yn_results):
            if is_yes:
                yes_entries.append((lid, plabel))

        if verbose:
            print(f"\nPointwise: {len(yes_entries)}/{len(self._candidates)} paths passed Yes/No filter")
            for (lid, plabel), (_, is_yes) in zip(self._candidates, yn_results):
                mark = "✓" if is_yes else "✗"
                print(f"  [{mark}] {plabel}")

        final_candidates = yes_entries if yes_entries else self._candidates
        fallback_used = len(yes_entries) == 0

        if len(final_candidates) == 1:
            selected_idx = 0
        else:
            selected_idx = await self._final_select(
                original_text, final_candidates, verbose=verbose
            )

        selected_lid, selected_path = final_candidates[selected_idx]
        selected_labels = self._labels_from_path_str(selected_path)

        details = {
            'method': 'pointwise',
            'llm_model': self.llm_model,
            'total_candidates': len(self._candidates),
            'yes_count': len(yes_entries),
            'fallback_used': fallback_used,
            'final_candidates': [plabel for _, plabel in final_candidates],
            'selected_leaf_id': selected_lid,
            'selected_leaf_path': selected_path,
            'final_prediction': ", ".join(selected_labels),
            'yn_results': [
                {'leaf': plabel, 'yes': is_yes}
                for (_, plabel), (_, is_yes) in zip(self._candidates, yn_results)
            ],
        }

        return selected_labels, details

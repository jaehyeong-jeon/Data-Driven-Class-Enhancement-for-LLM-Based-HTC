"""
Top-Down LLM Classifiers
========================
Two LLM-based top-down classifiers for hierarchical text classification.

TopDownLLMBeamSearch
--------------------
Beam search variant. Selection modes (--selection_mode):

  per_child  [default]
    Each beam independently ranks its own children → top-k per beam.
    Pool all results (up to B×k candidates) → final re-ranking → top-k.
    LLM calls per layer (after layer 0): B expansion calls + 1 pool re-rank.

  all_in_one
    All beams' children are gathered into ONE prompt → top-k selected directly.
    LLM calls per layer (after layer 0): 1.

  pointwise
    Gather ALL beams' children → yes/no per child (parallel) → filter "yes" → re-rank → top-k.
    LLM calls per layer (after layer 0): N_total_children + 1.

beam_size=1 is mechanistically equivalent to greedy top-down.

"""

import asyncio
import re

from src.utils import get_node_label, get_node_path


class TopDownLLMBeamSearch:

    def __init__(
        self,
        cfg,
        llm_model,
        beam_size=3,
        selection_mode='per_child',  # 'per_child' | 'all_in_one' | 'pointwise'
        stats=None,
        label2desc=None,
    ):
        self.llm_model      = llm_model
        self.id2label       = cfg["id2label"]
        self.layer2ids      = cfg["layer2ids"]
        self.id2info        = cfg["id2info"]
        self.dataset        = cfg["dataset"]
        self.num_layers     = cfg["num_layers"]
        self.id2paths       = cfg.get("id2paths")
        self.beam_size      = beam_size
        self.selection_mode = selection_mode
        self.stats          = stats
        self.label2desc     = label2desc or {}
        self.llm            = cfg["llm"]

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _children_of(self, beam_id, level):
        return [c for c in self.id2info[beam_id].get('children', [])
                if c in self.layer2ids[level]]

    def _path_label(self, node_id):
        if self.id2paths is not None:
            paths = self.id2paths.get(node_id, [])
            if paths:
                formatted = [p.replace(", ", " > ") for p in paths]
                return " / ".join(formatted)
        path = get_node_path(node_id, self.id2info)
        labels = [get_node_label(nid, self.id2label, self.dataset) for nid in path]
        return " > ".join(labels)

    def _desc_block(self, candidate_ids):
        lines = []
        for cid in candidate_ids:
            lbl = get_node_label(cid, self.id2label, self.dataset)
            desc = self.label2desc.get(lbl.lower(), "")
            if desc:
                lines.append(f"\nDescription for '{lbl}':\n  {desc}")
        return "\n".join(lines) if lines else ""

    # ── LLM primitives ────────────────────────────────────────────────────────

    async def _rank(self, text, candidate_ids, k, verbose=False):
        """Rank candidates, return top-k ids and raw response."""
        if len(candidate_ids) <= k:
            return candidate_ids, f"All {len(candidate_ids)} within k"

        from src.prompts import create_ranker_prompts
        candidates_info = [
            {'id': cid, 'label': get_node_label(cid, self.id2label, self.dataset)}
            for cid in candidate_ids
        ]
        system_prompt, user_prompt = create_ranker_prompts(
            text=text, candidates=candidates_info, dataset=self.dataset
        )
        if self.label2desc:
            desc_block = self._desc_block(candidate_ids)
            if desc_block:
                user_prompt += "\n\nCategory descriptions:" + desc_block

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]

        if verbose:
            print(f"\n=== Ranking Prompt ===\n{user_prompt}\n")

        response = await self.llm.acall(messages=messages, max_tokens=500, verbose=verbose)

        if verbose:
            print(f"\n=== Ranking Response ===\n{response}\n")

        ranked_ids = self._parse_ranking(response, candidate_ids, verbose=verbose)
        return ranked_ids[:k], response

    async def _ask_yn(self, text, node_id):
        """Yes/No relevance call for a single node. Returns (node_id, is_yes)."""
        from src.prompts import create_pointwise_yn_prompt
        prompt = create_pointwise_yn_prompt(text, self._path_label(node_id), self.dataset)
        
        if self.label2desc:
            lbl = get_node_label(node_id, self.id2label, self.dataset)
            desc = self.label2desc.get(lbl.lower(), "")
            if desc:
                prompt += f"\n\nCategory description for '{lbl}':\n  {desc}"


        response = await self.llm.acall(prompt=prompt, max_tokens=50, verbose=False)
        return node_id, "yes" in response.lower()

    def _parse_ranking(self, response, candidate_ids, verbose=False):
        n = len(candidate_ids)
        indices = [int(x) - 1 for x in re.findall(r'\d+', response)]
        indices = list(dict.fromkeys(indices))
        indices = [x for x in indices if 0 <= x < n] + [x for x in range(n) if x not in indices]
        ranked_ids = [candidate_ids[i] for i in indices]
        if verbose:
            print(f"Ranking: {[get_node_label(r, self.id2label, self.dataset) for r in ranked_ids]}")
        return ranked_ids

    # ── Main classify ─────────────────────────────────────────────────────────

    async def classify(self, text, verbose=False):
        original_text = text.replace("\n", " ").replace('"', '\\"')

        details = {
            'layers': [],
            'method': 'topdown_beamsearchsearch',
            'llm_model': self.llm_model,
            'beam_size': self.beam_size,
            'selection_mode': self.selection_mode,
        }

        current_paths = None

        for layer in range(self.num_layers):
            is_last = (layer == self.num_layers - 1)
            k = 1 if is_last else self.beam_size

            if verbose:
                beam_labels = [[get_node_label(n, self.id2label, self.dataset) for n in p]
                               for p in current_paths] if current_paths else []
                print(f"\n{'='*60}")
                print(f"Layer {layer} | beams={beam_labels} | k={k} | mode={self.selection_mode}")
                print(f"{'='*60}")

            layer_detail = {'layer': layer, 'beam_expansions': [], 'final_selection': None}

            # ── Layer 0: single expansion from root ──────────────────────────
            if current_paths is None:
                root_candidates = self.layer2ids[layer]

                if self.selection_mode == 'pointwise':
                    yn_results = await asyncio.gather(*[self._ask_yn(original_text, c) for c in root_candidates])
                    yes_ids = [nid for nid, is_yes in yn_results if is_yes]
                    to_rank = yes_ids if yes_ids else root_candidates
                    selected_ids, response = await self._rank(original_text, to_rank, k, verbose=verbose)
                    layer_detail['beam_expansions'].append({
                        'beam': None,
                        'candidates': [get_node_label(c, self.id2label, self.dataset) for c in root_candidates],
                        'yn_results': [{'label': get_node_label(nid, self.id2label, self.dataset), 'yes': is_yes}
                                       for nid, is_yes in yn_results],
                        'selected': [get_node_label(s, self.id2label, self.dataset) for s in selected_ids],
                        'llm_response': response,
                    })
                else:  # per_child, all_in_one
                    selected_ids, response = await self._rank(original_text, root_candidates, k, verbose=verbose)
                    layer_detail['beam_expansions'].append({
                        'beam': None,
                        'candidates': [get_node_label(c, self.id2label, self.dataset) for c in root_candidates],
                        'selected': [get_node_label(s, self.id2label, self.dataset) for s in selected_ids],
                        'llm_response': response,
                    })

                current_paths = [[sid] for sid in selected_ids]

            # ── all_in_one: gather ALL children → one prompt → top-k ─────────
            elif self.selection_mode == 'all_in_one':
                all_children = []
                child_to_parent = {}
                seen = set()
                for path in current_paths:
                    beam_id = path[-1]
                    for c in self._children_of(beam_id, layer):
                        child_to_parent[c] = beam_id
                        if c not in seen:
                            all_children.append(c)
                            seen.add(c)

                if not all_children:
                    if verbose:
                        print(f"No children at layer {layer}")
                    break

                if verbose:
                    print(f"all_in_one: {len(all_children)} total children across all beams")

                if is_last:
                    top_k_ids, response1 = await self._rank(original_text, all_children, self.beam_size, verbose=verbose)
                    final_ids, response2 = await self._rank(original_text, top_k_ids, 1, verbose=verbose)
                    layer_detail['final_selection'] = {
                        'pool': [get_node_label(c, self.id2label, self.dataset) for c in all_children],
                        'top_k': [get_node_label(s, self.id2label, self.dataset) for s in top_k_ids],
                        'selected': [get_node_label(s, self.id2label, self.dataset) for s in final_ids],
                        'llm_response1': response1,
                        'llm_response2': response2,
                    }
                else:
                    final_ids, response = await self._rank(original_text, all_children, k, verbose=verbose)
                    layer_detail['final_selection'] = {
                        'pool': [get_node_label(c, self.id2label, self.dataset) for c in all_children],
                        'selected': [get_node_label(s, self.id2label, self.dataset) for s in final_ids],
                        'llm_response': response,
                    }

                new_paths = []
                for sid in final_ids:
                    parent_id = child_to_parent.get(sid)
                    parent_path = next((p for p in current_paths if p[-1] == parent_id), current_paths[0])
                    new_paths.append(parent_path + [sid])
                current_paths = new_paths

            # ── pointwise: all beams' children → yes/no → filter → re-rank ───
            elif self.selection_mode == 'pointwise':
                all_children = []
                child_to_parent = {}
                seen = set()
                for path in current_paths:
                    beam_id = path[-1]
                    for c in self._children_of(beam_id, layer):
                        child_to_parent[c] = beam_id
                        if c not in seen:
                            all_children.append(c)
                            seen.add(c)

                if not all_children:
                    break

                yn_results = await asyncio.gather(*[self._ask_yn(original_text, c) for c in all_children])
                yes_ids = [nid for nid, is_yes in yn_results if is_yes]
                to_rank = yes_ids if yes_ids else all_children

                if verbose:
                    for nid, is_yes in yn_results:
                        mark = "✓" if is_yes else "✗"
                        print(f"  [{mark}] {get_node_label(nid, self.id2label, self.dataset)}")
                    print(f"\n--- Ranking {len(to_rank)} candidates ---")

                final_ids, final_response = await self._rank(original_text, to_rank, k, verbose=verbose)
                layer_detail['final_selection'] = {
                    'pool': [get_node_label(c, self.id2label, self.dataset) for c in all_children],
                    'yn_results': [{'label': get_node_label(nid, self.id2label, self.dataset), 'yes': is_yes}
                                   for nid, is_yes in yn_results],
                    'selected': [get_node_label(s, self.id2label, self.dataset) for s in final_ids],
                    'llm_response': final_response,
                }

                new_paths = []
                for sid in final_ids:
                    parent_id = child_to_parent.get(sid)
                    parent_path = next((p for p in current_paths if p[-1] == parent_id), current_paths[0])
                    new_paths.append(parent_path + [sid])
                current_paths = new_paths

            # ── per_child: each beam ranks its own children → pool → re-rank ─
            else:
                child_to_parent = {}
                pool = []
                seen = set()

                async def expand_beam(path):
                    beam_id = path[-1]
                    children = self._children_of(beam_id, layer)
                    if not children:
                        return beam_id, children, "No children"
                    top_children, resp = await self._rank(original_text, children, k, verbose=verbose)
                    return beam_id, top_children, resp

                results = await asyncio.gather(*[expand_beam(path) for path in current_paths])

                for path, (beam_id, top_children, response) in zip(current_paths, results):
                    beam_label = get_node_label(beam_id, self.id2label, self.dataset)
                    if not top_children:
                        if verbose:
                            print(f"  Beam '{beam_label}' has no children at layer {layer}")
                        continue
                    layer_detail['beam_expansions'].append({
                        'beam': beam_label,
                        'selected': [get_node_label(s, self.id2label, self.dataset) for s in top_children],
                        'llm_response': response,
                    })
                    for cid in top_children:
                        child_to_parent[cid] = beam_id
                        if cid not in seen:
                            seen.add(cid)
                            pool.append(cid)

                if not pool:
                    if verbose:
                        print(f"No candidates in pool at layer {layer}")
                    break

                if verbose:
                    print(f"\n--- Final selection from pool of {len(pool)} ---")

                final_ids, final_response = await self._rank(original_text, pool, k, verbose=verbose)
                layer_detail['final_selection'] = {
                    'pool': [get_node_label(c, self.id2label, self.dataset) for c in pool],
                    'selected': [get_node_label(s, self.id2label, self.dataset) for s in final_ids],
                    'llm_response': final_response,
                }

                new_paths = []
                for sid in final_ids:
                    parent_id = child_to_parent.get(sid)
                    parent_path = next((p for p in current_paths if p[-1] == parent_id), current_paths[0])
                    new_paths.append(parent_path + [sid])
                current_paths = new_paths

            details['layers'].append(layer_detail)

        final_path = current_paths[0]
        selected_labels = [get_node_label(nid, self.id2label, self.dataset) for nid in final_path]
        details['final_prediction'] = ", ".join(selected_labels)
        details['final_node_path'] = final_path

        return selected_labels, details

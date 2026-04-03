"""
Path Classifier
===============
Lists all root→leaf taxonomy paths in a single prompt and asks the LLM
to pick one.  Equivalent to the 'path' method in main_amazon/dbpedia/wos.py,
generalized to all three datasets.

One LLM call per sample.
"""

from src.utils import get_node_label, get_node_path
from src.prompts import PromptManager


class PathClassifier:
    """
    Flat path-based classifier: list every root→leaf path, LLM picks one.

    Matches the 'path single' method from main_amazon/dbpedia/wos exactly:
    - Same prompt template (USER_PATH[dataset]['single'])
    - Same two-message format (instruction + input_text as separate user messages)
    - Same path string format per dataset
    """

    def __init__(
        self,
        cfg,
        llm_model,
    ):
        self.llm        = cfg["llm"]
        self.llm_model  = llm_model
        self.id2label   = cfg["id2label"]
        self.layer2ids  = cfg["layer2ids"]
        self.id2info    = cfg["id2info"]
        self.dataset    = cfg["dataset"]
        self.num_layers = cfg["num_layers"]
        self.id2paths   = cfg.get("id2paths")  # None for dbpedia/wos (use get_node_path)

        self._pm = PromptManager(self.dataset)

        # Pre-build (leaf_id, path_string) pairs once
        leaf_layer = self.num_layers - 1
        self.leaf_ids = self.layer2ids[leaf_layer]
        self.path_strings = self._build_path_strings()

    # ── Path string construction ───────────────────────────────────────────────

    def _build_path_strings(self):
        """Build path strings per leaf, matching each dataset's native format.

        For Amazon: uses id2paths which lists ALL paths for multi-parent nodes,
                    exactly like main_amazon.py does.
        For others: reconstructs the single path via get_node_path.
        """
        if self.id2paths is not None:
            # Amazon: each leaf may have multiple paths (multi-parent nodes)
            # main_amazon.py does: candidates += id2paths.get(lid, [])
            # so path_strings is a flat list of all path strings across all leaves
            path_strings = []
            self.leaf_id_per_path = []  # track which leaf each path string belongs to
            for lid in self.leaf_ids:
                paths = self.id2paths.get(lid, [])
                if not paths:
                    p = ", ".join(get_node_label(nid, self.id2label, self.dataset)
                                  for nid in get_node_path(lid, self.id2info))
                    path_strings.append(p)
                    self.leaf_id_per_path.append(lid)
                else:
                    for p in paths:
                        path_strings.append(p)
                        self.leaf_id_per_path.append(lid)
            return path_strings
        else:
            self.leaf_id_per_path = list(self.leaf_ids)
            path_strings = []
            for lid in self.leaf_ids:
                path_ids = get_node_path(lid, self.id2info)
                labels = [get_node_label(nid, self.id2label, self.dataset) for nid in path_ids]
                path_strings.append(", ".join(labels))
            return path_strings

    # ── Prompt building ────────────────────────────────────────────────────────

    def _build_prompt(self, text):
        candidates_str = "; ".join(self.path_strings)
        user_prompt = self._pm.path_prompt('single', candidates_str)
        return user_prompt, text

    # ── Response parsing ───────────────────────────────────────────────────────

    def _parse_response_to_labels(self, response):
        """Parse LLM response into per-layer labels using the same approach as main_amazon.py.

        Instead of trying to match against full path strings and then reconstructing
        via get_node_path (which fails for multi-parent nodes and defaults incorrectly),
        we directly use the raw LLM output split by ', ' and match per-layer against
        per-layer candidate names.
        """
        from src.utils import post_process, get_result

        cleaned = post_process(response.strip())
        # Remove brackets that GPT sometimes adds
        cleaned = cleaned.replace('[', '').replace(']', '').strip()
        parts = [p.strip() for p in cleaned.split(', ')]

        # Build per-layer candidate names (same as main_amazon.py evaluation)
        candidates_per_layer = [
            [get_node_label(nid, self.id2label, self.dataset)
             for nid in self.layer2ids[l]]
            for l in range(self.num_layers)
        ]

        selected_labels = []
        for layer in range(self.num_layers):
            if len(parts) == self.num_layers:
                matched = get_result(parts[layer], candidates_per_layer[layer])
            else:
                # Fallback: try matching entire string against this layer's candidates
                matched = get_result(', '.join(parts), candidates_per_layer[layer])
            selected_labels.append(matched if matched != 'none_class' else 'none_class')

        return selected_labels

    # ── Main classify ──────────────────────────────────────────────────────────

    def classify(self, text, verbose=False, max_retries=3):
        """
        Returns:
            selected_labels: list of labels root→leaf
            details: dict for logging
        """
        text = text.replace("\n", " ").replace('"', '\\"')
        user_prompt, input_text = self._build_prompt(text)

        if verbose:
            print(f"\n=== Path Prompt ===\n{user_prompt}\n{input_text}\n")

        attempts = []
        for attempt in range(max_retries):
            response = self.llm.call(prompt=user_prompt, input_text=input_text, verbose=(verbose and attempt == 0))

            if verbose:
                print(f"\n=== LLM Response (attempt {attempt+1}) ===\n{response}\n")

            selected_labels = self._parse_response_to_labels(response)
            attempts.append({'attempt': attempt + 1, 'llm_response': response,
                             'matched': ", ".join(selected_labels)})

            if 'none_class' not in selected_labels:
                break

        details = {
            'method': 'path',
            'llm_model': self.llm_model,
            'llm_response': response,
            'final_prediction': ", ".join(selected_labels),
            'attempts': attempts,
        }

        return selected_labels, details

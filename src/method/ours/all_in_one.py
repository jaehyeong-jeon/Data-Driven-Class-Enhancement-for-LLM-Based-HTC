"""
All-In-One Classifier
=====================
Single LLM call baseline: present all taxonomy paths in a single prompt
and ask the LLM to pick one.
"""

from src.utils import get_node_label, get_node_path
from src.prompts import PromptManager


class AllInOneClassifier:
    """
    All-in-one classifier: list every root→leaf path, LLM picks one.
    """

    def __init__(
        self,
        cfg,
        llm_model,
        label2desc=None,
    ):
        self.llm_model  = llm_model
        self.llm        = cfg["llm"]
        self.id2label   = cfg["id2label"]
        self.layer2ids  = cfg["layer2ids"]
        self.id2info    = cfg["id2info"]
        self.dataset    = cfg["dataset"]
        self.num_layers = cfg["num_layers"]
        self.id2paths   = cfg.get("id2paths")
        self.label2desc = label2desc or {}
        
        self.keyword2class = {}
        if self.label2desc:
            for class_name, desc_str in self.label2desc.items():
                for line in desc_str.split("\n"):
                    line = line.strip()
                    if line.lower().startswith("core keywords:"):
                        for kw in line.split(":", 1)[1].split(","):
                            kw = kw.strip().lower()
                            if kw and kw != class_name.lower():
                                self.keyword2class[kw] = class_name.lower()

        self._pm = PromptManager(self.dataset)

        leaf_layer = self.num_layers - 1
        self.leaf_ids = self.layer2ids[leaf_layer]
        self.path_strings = self._build_path_strings()

    def _build_path_strings(self):
        """Build path strings per leaf."""
        if self.id2paths is not None:
            path_strings = []
            self.leaf_id_per_path = []
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

    def _build_prompt(self, text):
        candidates_str = "; ".join(self.path_strings)
        user_prompt = self._pm.path_prompt('single', candidates_str)
        if self.label2desc:
            user_prompt += "\n\nBelow are descriptions for each taxonomy path:\n"
            for path_str in self.path_strings:
                parts = [p.strip() for p in path_str.split(", ")]
                user_prompt += f"\n[{path_str}]\n"
                for part in parts:
                    desc = self.label2desc.get(part.lower(), "")
                    if desc:
                        user_prompt += f"  ({part}): {desc}\n"
            user_prompt += (
                f"\nYou MUST choose exactly one from the following paths: "
                f"[{'; '.join(self.path_strings)}]. "
                f"Answer with the complete path from root to leaf, separated by ', '. "
                f"Do NOT output keywords, descriptions, or any text other than the exact path. "
            )
        else:
            user_prompt += (
                "You MUST answer with the complete path from root to leaf, "
                "separated by ', '. Do NOT omit any level. "
            )
        return user_prompt, text

    def _parse_response_to_labels(self, response):
        """Parse LLM response into per-layer labels."""
        from src.utils import post_process, get_result

        cleaned = post_process(response.strip())
        cleaned = cleaned.replace('[', '').replace(']', '').strip()
        parts = [p.strip() for p in cleaned.split(', ')]

        candidates_per_layer = [
            [get_node_label(nid, self.id2label, self.dataset)
             for nid in self.layer2ids[l]]
            for l in range(self.num_layers)
        ]

        if len(parts) < self.num_layers:
            leaf_layer = self.num_layers - 1
            leaf_match = get_result(', '.join(parts), candidates_per_layer[leaf_layer])
            if leaf_match and leaf_match != 'none_class':
                leaf_id = None
                for nid in self.layer2ids[leaf_layer]:
                    if get_node_label(nid, self.id2label, self.dataset) == leaf_match:
                        leaf_id = nid
                        break
                if leaf_id is not None:
                    path_labels = [None] * self.num_layers
                    path_labels[leaf_layer] = leaf_match
                    cur = leaf_id
                    for layer in range(leaf_layer - 1, -1, -1):
                        parents = self.id2info.get(cur, {}).get('parent')
                        if parents:
                            cur = parents[0] if isinstance(parents, list) else parents
                            path_labels[layer] = get_node_label(cur, self.id2label, self.dataset)
                        else:
                            break
                    if all(pl is not None for pl in path_labels):
                        return path_labels

        selected_labels = []
        for layer in range(self.num_layers):
            if len(parts) == self.num_layers:
                matched = get_result(parts[layer], candidates_per_layer[layer])
            else:
                matched = get_result(", ".join(parts), candidates_per_layer[layer])
            selected_labels.append(matched if matched != 'none_class' else 'none_class')

        if 'none_class' in selected_labels and self.keyword2class:
            for layer in range(self.num_layers):
                if selected_labels[layer] != 'none_class':
                    continue
                candidates_lower = {c.lower() for c in candidates_per_layer[layer]}
                best_match, best_count = None, 0
                for kw, cn in self.keyword2class.items():
                    if cn in candidates_lower and kw in cleaned:
                        count = cleaned.count(kw)
                        if count > best_count:
                            best_count = count
                            best_match = cn
                if best_match:
                    selected_labels[layer] = best_match

        return selected_labels

    def classify(self, text, verbose=False, max_retries=3):
        text = text.replace("\n", " ").replace('"', '\\"')
        user_prompt, input_text = self._build_prompt(text)

        attempts = []
        for attempt in range(max_retries):
            response = self.llm.call(prompt=user_prompt, input_text=input_text, verbose=(verbose and attempt == 0))
            selected_labels = self._parse_response_to_labels(response)
            attempts.append({'attempt': attempt + 1, 'llm_response': response,
                             'matched': ", ".join(selected_labels)})

            if 'none_class' not in selected_labels:
                break

        details = {
            'method': 'all_in_one',
            'llm_model': self.llm_model,
            'llm_response': response,
            'final_prediction': ", ".join(selected_labels),
            'attempts': attempts,
        }

        return selected_labels, details

from src.utils import read_pickle

DATA_ROOT = 'dataset'


def load_dataset_config(dataset_name):
    dataset_name = dataset_name.lower()
    prefix = dataset_name.upper()
    path = f'{DATA_ROOT}/{dataset_name}/{prefix}'

    keys = ['hierarchy', 'hierarchy_simplify', 'id2info', 'layer2ids', 'label2id', 'id2label']
    if dataset_name == 'amazon':
        keys.append('id2paths')
    else:
        keys.append('id2path')

    config = {}
    for key in keys:
        var_name = f'{prefix}_{key}'
        try:
            config[var_name] = read_pickle(f'{path}_{key}.pkl')
        except Exception:
            config[var_name] = {}

    return config


# Load all datasets on import
globals().update(load_dataset_config('amazon'))
globals().update(load_dataset_config('dbpedia'))
globals().update(load_dataset_config('wos'))

import os
import json

NUM_LAYERS = {"wos": 2, "amazon": 3, "dbpedia": 3}

DESC_PATHS = {
    "wos":    "dataset/wos/description.json",
    "amazon": "dataset/amazon/description.json",
    "dbpedia": "dataset/dbpedia/description.json",
}
_KEY_NORM = {"mental_health": "mental health"}

def load_label2desc(dataset) -> dict:
    """label(lowercase) -> description string (from build_descriptions.py output)."""
    desc_path = DESC_PATHS.get(dataset, "")
    if not os.path.exists(desc_path):
        return {}
    with open(desc_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    leaf_layer = NUM_LAYERS[dataset] - 1
    leaf_data = data.get(dataset, {}).get(f"layer {leaf_layer}", {})
    label2desc = {}
    for raw_key, d in leaf_data.items():
        label = _KEY_NORM.get(raw_key, raw_key)
        lines = [f"Definition: {d['definition']}"]
        lines.append("Characteristics: " + " / ".join(d["characteristics"]))
        lines.append("Core keywords: " + ", ".join(d["core_keywords"]))
        if "classification_criteria" in d:
            cr = d["classification_criteria"]
            lines.append(f"Unit of analysis: {cr['Unit_of_Analysis']}")
            lines.append(f"Primary goal: {cr['Primary_Goal']}")
            lines.append(f"Approach type: {cr['Approach_Type']}")
        label2desc[label] = "\n  ".join(lines)
    return label2desc


def load_id2desc(dataset, id2label):
    """Load old-format descriptions (data/{dataset}/{PREFIX}_descriptions.json) keyed by node id."""
    prefix = dataset.upper()
    path = f"dataset/{dataset}/{prefix}_descriptions.json"
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {int(k): v["description"] for k, v in data.items()}

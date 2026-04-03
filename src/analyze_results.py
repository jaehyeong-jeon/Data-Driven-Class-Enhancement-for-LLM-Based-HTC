"""
JSON 결과 파일 분석 스크립트
"""
import json
import pandas as pd
from collections import Counter


def load_jsonl(file_path):
    """Load JSONL file and return list of records."""
    results = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            results.append(json.loads(line))
    return results


def analyze_classification_process(jsonl_file):
    """Analyze classification process from JSONL file."""
    results = load_jsonl(jsonl_file)

    print(f"Total samples: {len(results)}")
    print(f"\n{'='*60}")

    # Example: Print first sample's classification process
    if len(results) > 0:
        sample = results[0]
        print(f"\nSample {sample['index']}:")
        print(f"Text: {sample['text']}")
        print(f"Final Prediction: {sample['prediction']}")
        print(f"\nClassification Process:")

        for layer_info in sample['classification_process']['layers']:
            print(f"\n  Layer {layer_info['layer']}:")
            print(f"    Active nodes: {layer_info['active_node_labels']}")

            if 'local_search' in layer_info:
                print(f"    Local search:")
                for parent_search in layer_info['local_search']:
                    print(f"      Parent: {parent_search['parent_label']}")
                    print(f"      Children: {parent_search['children'][:5]}...")  # Show first 5

                print(f"    Global search:")
                print(f"      All candidates: {layer_info['global_search']['all_candidates'][:5]}...")

            print(f"    Selected: {layer_info['selected_node_labels']}")
            print(f"    Prediction: {layer_info['prediction']}")

    print(f"\n{'='*60}")

    # Statistics
    predictions = [r['prediction'] for r in results]
    errors = [r for r in results if r['prediction'] == 'ERROR']

    print(f"\nStatistics:")
    print(f"  Success: {len(results) - len(errors)}")
    print(f"  Errors: {len(errors)}")

    if len(errors) > 0:
        print(f"\nError samples: {[e['index'] for e in errors]}")

    # Top predictions
    pred_counter = Counter([p for p in predictions if p != 'ERROR'])
    print(f"\nTop 10 predictions:")
    for pred, count in pred_counter.most_common(10):
        print(f"  {pred}: {count}")


def export_to_csv(jsonl_file, output_csv):
    """Export predictions to CSV for easy comparison."""
    results = load_jsonl(jsonl_file)

    data = []
    for r in results:
        row = {
            'index': r['index'],
            'text': r['text'],
            'prediction': r['prediction']
        }

        # Add layer-wise predictions
        if 'classification_process' in r:
            for i, layer in enumerate(r['classification_process']['layers']):
                row[f'layer_{i}_prediction'] = layer.get('prediction', '')
                row[f'layer_{i}_selected'] = ', '.join(layer.get('selected_node_labels', []))

        data.append(row)

    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"Exported to {output_csv}")


def compare_local_global_search(jsonl_file, layer=1):
    """Compare local vs global search results at a specific layer."""
    results = load_jsonl(jsonl_file)

    print(f"Analyzing Layer {layer} search process...")

    local_selected = []
    global_selected = []

    for r in results:
        if 'classification_process' not in r:
            continue

        layers = r['classification_process']['layers']
        if len(layers) <= layer:
            continue

        layer_info = layers[layer]

        if 'local_search' in layer_info:
            # Collect all locally selected candidates
            for parent_search in layer_info['local_search']:
                if 'children' in parent_search and 'selected_indices' in parent_search:
                    selected = [parent_search['children'][i] for i in parent_search['selected_indices']]
                    local_selected.extend(selected)

            # Collect globally selected candidates
            if 'global_search' in layer_info:
                global_search = layer_info['global_search']
                if 'all_candidates' in global_search and 'selected_indices' in global_search:
                    selected = [global_search['all_candidates'][i] for i in global_search['selected_indices']]
                    global_selected.extend(selected)

    print(f"\nLocal search selected {len(local_selected)} candidates in total")
    print(f"Global search selected {len(global_selected)} candidates in total")

    local_counter = Counter(local_selected)
    global_counter = Counter(global_selected)

    print(f"\nTop 10 locally selected categories:")
    for cat, count in local_counter.most_common(10):
        print(f"  {cat}: {count}")

    print(f"\nTop 10 globally selected categories:")
    for cat, count in global_counter.most_common(10):
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python analyze_results.py <jsonl_file>")
        print("Example: python analyze_results.py results/gpt-4o-mini/wos_eval5k0,100_ours_k3-3_details.jsonl")
        sys.exit(1)

    jsonl_file = sys.argv[1]

    # Analyze classification process
    analyze_classification_process(jsonl_file)

    # Export to CSV
    output_csv = jsonl_file.replace('_details.jsonl', '_predictions.csv')
    export_to_csv(jsonl_file, output_csv)

    # Compare local vs global search
    print(f"\n{'='*60}")
    compare_local_global_search(jsonl_file, layer=1)

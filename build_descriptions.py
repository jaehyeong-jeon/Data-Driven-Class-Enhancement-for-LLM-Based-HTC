"""
Build category descriptions for hierarchical datasets.

Process:
1. (Leaf) For each leaf node, sample up to --sample_n documents, extract
   keywords/summary per doc via LLM, then synthesize into
   {definition, characteristics, core_keywords}.
2. (Parent) Bottom-up: synthesize each parent's description from its
   children's descriptions.
3. (Discriminative) For each class, find embedding-similar neighbours and
   generate a detail_description that distinguishes it from them.

Usage:
    python build_descriptions.py --dataset wos     --model gpt-4o-mini
    python build_descriptions.py --dataset amazon  --model gpt-4o-mini
    python build_descriptions.py --dataset dbpedia --model gpt-4o-mini

    # Resume interrupted run (skips already-done entries)
    python build_descriptions.py --dataset wos --model gpt-4o-mini --resume

    # Only redo parent / discriminative steps
    python build_descriptions.py --dataset wos --model gpt-4o-mini --parents_only
    python build_descriptions.py --dataset wos --model gpt-4o-mini --discriminative_only
"""

import argparse
import asyncio
import json
import os
import random
import re

import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm import tqdm

from config import load_dataset_config, DESC_PATHS

load_dotenv()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clean_for_save(result):
    """Remove internal fields (prefixed with _) before saving to JSON."""
    cleaned = {}
    for layer_key, classes in result.items():
        cleaned[layer_key] = {}
        for label, desc in classes.items():
            if isinstance(desc, dict):
                cleaned[layer_key][label] = {k: v for k, v in desc.items() if not k.startswith('_')}
            else:
                cleaned[layer_key][label] = desc
    return cleaned


# ── Data loading ──────────────────────────────────────────────────────────────

def load_dataset_docs(dataset):
    """Load documents with leaf-level labels from train/val splits (not eval5k).

    Returns:
        df: DataFrame with columns ['text', 'leaf_label']
    """
    if dataset == 'wos':
        sample_path = 'dataset/wos/Data_sample.csv'
        full_path   = 'dataset/wos/Data.xlsx'
        if os.path.exists(sample_path):
            df = pd.read_csv(sample_path, usecols=['area', 'Abstract'])
        else:
            df = pd.read_excel(full_path, usecols=['area', 'Abstract'])
        df['leaf_label'] = df['area'].str.strip().str.lower()
        return df.rename(columns={'Abstract': 'text'})[['text', 'leaf_label']]

    elif dataset == 'amazon':
        sample_path = 'dataset/amazon/train_sample.csv'
        if os.path.exists(sample_path):
            d = pd.read_csv(sample_path, usecols=['Text', 'Cat3'])
            d = d.rename(columns={'Text': 'text', 'Cat3': 'leaf_label'})
            d = d[d['leaf_label'] != 'unknown']
            d['leaf_label'] = '2-' + d['leaf_label']
            return d[['text', 'leaf_label']]
        dfs = []
        for split, col in [('train_40k.csv', 'Cat3'), ('val_10k.csv', 'Cat3')]:
            d = pd.read_csv(f'dataset/amazon/{split}', usecols=['Text', col])
            d = d.rename(columns={'Text': 'text', col: 'leaf_label'})
            dfs.append(d)
        df = pd.concat(dfs, ignore_index=True)
        df = df[df['leaf_label'] != 'unknown']
        df['leaf_label'] = '2-' + df['leaf_label']
        return df[['text', 'leaf_label']]

    elif dataset == 'dbpedia':
        def camel_to_lower(s):
            s = re.sub(r'([a-z])([A-Z])', r'\1 \2', s)
            s = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', s)
            return s.lower()

        sample_path = 'dataset/dbpedia/DBPEDIA_train_sample.csv'
        if os.path.exists(sample_path):
            df = pd.read_csv(sample_path, usecols=['text', 'l3'])
            df = df.rename(columns={'l3': 'leaf_label'})
        else:
            dfs = []
            for split in ['DBPEDIA_train.csv', 'DBPEDIA_val.csv', 'DBPEDIA_test.csv']:
                path = f'dataset/dbpedia/{split}'
                if os.path.exists(path):
                    d = pd.read_csv(path, usecols=['text', 'l3'])
                    dfs.append(d.rename(columns={'l3': 'leaf_label'}))
            df = pd.concat(dfs, ignore_index=True)
        df['leaf_label'] = df['leaf_label'].apply(camel_to_lower)
        df['leaf_label'] = df['leaf_label'].replace('ncaa team season', 'ncaateam season')
        return df[['text', 'leaf_label']]

    else:
        raise ValueError(f"Unknown dataset: {dataset}")


# ── Prompts ───────────────────────────────────────────────────────────────────

EXTRACT_PROMPT = """You are analyzing a document.

Document:
\"\"\"
{text}
\"\"\"

Extract the following from this document:
1. **keywords**: A list of 5-8 key terms/phrases that characterize this document's topic. Prefer exact terms and phrases as they appear in the original text.
2. **summary**: A 1-2 sentence summary capturing the main subject and contribution. Use the original document's own wording, terminology, and phrasing as much as possible — avoid paraphrasing or substituting with generic terms.

Respond in JSON format:
{{"keywords": ["kw1", "kw2", ...], "summary": "..."}}"""

SYNTHESIZE_PROMPT = """You are analyzing a collection of {n_docs} documents to identify what kind of topic area they collectively represent.

Below are keywords and summaries extracted from each document.

Combined Keywords:
{all_keywords}

Document Summaries:
{all_summaries}

Your task is NOT to summarize individual documents or pick specific details from them.
Instead, step back and think: "What general field or topic do ALL of these documents belong to?"

Generate the following based on the shared, recurring patterns across the entire collection:
1. **definition**: A 1-2 sentence description of the general topic area these documents collectively belong to. Focus on what is universally true across all documents, not specific to any single one.
2. **characteristics**: A list of 3-5 general characteristics that define this topic area as a whole — traits that would apply to ANY document in this domain, not just the ones shown.
3. **core_keywords**: A list of 5-8 abstract, high-level keywords that represent the domain itself. Avoid overly specific terms that only apply to a subset of documents; prefer terms that would generalize across the entire field.

Respond in JSON format:
{{"definition": "...", "characteristics": ["...", "..."], "core_keywords": ["...", "..."]}}"""

PARENT_SYNTHESIZE_PROMPT = """You are analyzing a group of related categories in a hierarchical taxonomy to identify the broader domain they all belong to.

The following child categories are grouped together under the same parent:

{children_info}

Your task is to generate a description for the PARENT category that encompasses ALL of these children.
Think: "What broader concept or field unifies all of these child categories?"

The description should:
- Be general enough to cover ALL child categories, not biased toward any single one.
- Capture the overarching theme that connects all children.
- Use abstract, high-level terms rather than specifics from any single child.

Generate the following:
1. **definition**: A 1-2 sentence description of this broad parent category that would logically contain all the listed child categories.
2. **characteristics**: A list of 3-5 general characteristics shared across all child categories at a high level.
3. **core_keywords**: A list of 5-8 high-level keywords that represent the parent domain as a whole.

Respond in JSON format:
{{"definition": "...", "characteristics": ["...", "..."], "core_keywords": ["...", "..."]}}"""

DISCRIMINATIVE_PROMPT = """You are refining a category description to make it more distinguishable from similar categories.

Here is the target category and its current description:

[Target Category]
Definition: {target_definition}
Characteristics: {target_characteristics}
Core keywords: {target_keywords}

Example document summaries from this category:
{target_doc_summaries}

The following categories are highly similar and could easily be confused with the target:

{similar_classes_info}

Your task is to generate a **detail_description** that clearly explains what makes the target category DISTINCT from each of these similar categories.

For each similar category, identify the specific boundary or criterion that separates it from the target. Be concrete and precise — avoid vague statements like "they are different fields." Focus on the actual content, methodology, and subject matter differences.

Generate the following:
1. **detail_description**: A 2-4 sentence explanation of what uniquely defines the target category and distinguishes it from ALL of the similar categories listed above. Focus on the core traits, methods, or subject matter that only the target category has, not shared by any of the similar ones.

Respond in JSON format:
{{"detail_description": "..."}}"""


# ── LLM calls ────────────────────────────────────────────────────────────────

async def call_llm(client, model, prompt, max_retries=5):
    for attempt in range(max_retries):
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"  FAILED after {max_retries} attempts: {e}")
                return None
            await asyncio.sleep(0.5 * (attempt + 1))


async def extract_from_doc(client, model, text):
    return await call_llm(client, model, EXTRACT_PROMPT.format(text=text))


async def synthesize_description(client, model, extractions):
    all_keywords, all_summaries = [], []
    for i, ext in enumerate(extractions, 1):
        if ext is None:
            continue
        all_keywords.extend(ext.get('keywords', []))
        all_summaries.append(f"{i}. {ext.get('summary', 'N/A')}")
    prompt = SYNTHESIZE_PROMPT.format(
        n_docs=len(extractions),
        all_keywords=', '.join(all_keywords),
        all_summaries='\n'.join(all_summaries),
    )
    return await call_llm(client, model, prompt)


# ── Pipeline ──────────────────────────────────────────────────────────────────

async def process_leaf(client, model, leaf_label, docs_df, sample_n=10):
    """Sample docs for a leaf node → extract → synthesize."""
    leaf_docs = docs_df[docs_df['leaf_label'] == leaf_label]
    if len(leaf_docs) == 0:
        print(f"  WARNING: No documents found for '{leaf_label}', skipping.")
        return None

    sampled = leaf_docs.sample(n=min(sample_n, len(leaf_docs)), random_state=42)
    extractions = await asyncio.gather(*[
        extract_from_doc(client, model, row['text']) for _, row in sampled.iterrows()
    ])
    valid = [e for e in extractions if e is not None]
    if not valid:
        print(f"  WARNING: All extractions failed for '{leaf_label}'.")
        return None

    result = await synthesize_description(client, model, valid)
    if result is not None:
        result['_doc_summaries'] = [e.get('summary', '') for e in valid if e.get('summary')]
    return result


async def process_parent(client, model, parent_label, children_descs):
    """Synthesize parent description from children descriptions."""
    parts = []
    for child_label, child_desc in children_descs.items():
        lines = [f"[{child_label}]"]
        if isinstance(child_desc, dict):
            if 'definition' in child_desc:
                lines.append(f"  Definition: {child_desc['definition']}")
            if 'characteristics' in child_desc:
                lines.append(f"  Characteristics: {' / '.join(child_desc['characteristics'])}")
            if 'core_keywords' in child_desc:
                lines.append(f"  Core keywords: {', '.join(child_desc['core_keywords'])}")
        parts.append('\n'.join(lines))
    prompt = PARENT_SYNTHESIZE_PROMPT.format(children_info='\n\n'.join(parts))
    return await call_llm(client, model, prompt)


def find_similar_classes(descriptions, model_name="all-mpnet-base-v2", threshold=0.5, drop_ratio=0.8):
    """Find confusable classes per level using sentence-embedding cosine similarity."""
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    def desc_to_text(entry):
        parts = [entry.get("definition", "")]
        chars = entry.get("characteristics", [])
        if isinstance(chars, list):
            parts.append(". ".join(chars))
        keywords = entry.get("core_keywords", [])
        if isinstance(keywords, list):
            parts.append("Keywords: " + ", ".join(keywords))
        return " ".join(parts)

    labels = sorted(descriptions.keys())
    embeddings = SentenceTransformer(model_name).encode(
        [desc_to_text(descriptions[l]) for l in labels], show_progress_bar=False
    )
    sim_matrix = cosine_similarity(embeddings)

    similar_map = {}
    for i, label in enumerate(labels):
        sims = sim_matrix[i].copy()
        sims[i] = -1
        selected, top1 = [], None
        for j in np.argsort(sims)[::-1]:
            score = sims[j]
            if score < threshold:
                break
            if top1 is None:
                top1 = score
            elif score < top1 * drop_ratio:
                break
            selected.append((labels[j], float(score)))
        similar_map[label] = selected
    return similar_map


async def generate_discriminative(client, model, target_label, target_desc, similar_descs):
    """Generate detail_description distinguishing target from similar classes."""
    similar_parts = []
    for idx, (sim_label, sim_desc, sim_score) in enumerate(similar_descs, 1):
        lines = [f"[Similar Category {idx}]"]
        if isinstance(sim_desc, dict):
            if 'definition' in sim_desc:
                lines.append(f"  Definition: {sim_desc['definition']}")
            if 'characteristics' in sim_desc:
                lines.append(f"  Characteristics: {' / '.join(sim_desc['characteristics'])}")
            if 'core_keywords' in sim_desc:
                lines.append(f"  Core keywords: {', '.join(sim_desc['core_keywords'])}")
        similar_parts.append('\n'.join(lines))

    chars = target_desc.get('characteristics', [])
    keywords = target_desc.get('core_keywords', [])
    doc_summaries = target_desc.get('_doc_summaries', []) or target_desc.get('doc_summaries', [])
    summaries_text = (
        '\n'.join(f"  - {s}" for s in doc_summaries)
        if doc_summaries else '  (no document summaries available)'
    )
    prompt = DISCRIMINATIVE_PROMPT.format(
        target_definition=target_desc.get('definition', ''),
        target_characteristics=' / '.join(chars) if isinstance(chars, list) else str(chars),
        target_keywords=', '.join(keywords) if isinstance(keywords, list) else str(keywords),
        target_doc_summaries=summaries_text,
        similar_classes_info='\n\n'.join(similar_parts),
    )
    return await call_llm(client, model, prompt)


# ── Entry point ───────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',             required=True, choices=['wos', 'amazon', 'dbpedia'])
    parser.add_argument('--model',               default='gpt-4o-mini')
    parser.add_argument('--sample_n',            type=int,   default=10)
    parser.add_argument('--concurrency',         type=int,   default=20)
    parser.add_argument('--seed',                type=int,   default=42)
    parser.add_argument('--resume',              action='store_true')
    parser.add_argument('--parents_only',        action='store_true')
    parser.add_argument('--discriminative_only', action='store_true')
    parser.add_argument('--sim_threshold',       type=float, default=0.5)
    parser.add_argument('--sim_drop_ratio',      type=float, default=0.8)
    args = parser.parse_args()

    random.seed(args.seed)

    prefix = args.dataset.upper()
    raw       = load_dataset_config(args.dataset)
    layer2ids = raw[f'{prefix}_layer2ids']
    id2label  = raw[f'{prefix}_id2label']
    id2info   = raw[f'{prefix}_id2info']

    num_layers = len(layer2ids)
    leaf_layer = num_layers - 1
    leaf_ids   = layer2ids[leaf_layer]
    out_path   = DESC_PATHS[args.dataset]

    print(f"Dataset: {args.dataset} | Layers: {num_layers} | Leaf nodes: {len(leaf_ids)} | Model: {args.model}")

    existing = {}
    if (args.resume or args.parents_only or args.discriminative_only) and os.path.exists(out_path):
        with open(out_path, 'r') as f:
            existing = json.load(f)
        print(f"Loaded existing: {sum(len(v) for v in existing.get(args.dataset, {}).values())} entries")

    result = existing.get(args.dataset, {})
    for layer_idx in range(num_layers):
        result.setdefault(f"layer {layer_idx}", {})

    leaf_layer_key = f"layer {leaf_layer}"
    client = AsyncOpenAI()

    # ── Step 1: Leaf descriptions ─────────────────────────────────────────────
    if not args.parents_only and not args.discriminative_only:
        print("\nLoading documents...")
        docs_df    = load_dataset_docs(args.dataset)
        doc_labels = set(docs_df['leaf_label'].unique())
        leaf_labels = set(id2label[lid] for lid in leaf_ids)
        print(f"Documents: {len(docs_df)} | Leaf coverage: {len(leaf_labels & doc_labels)}/{len(leaf_labels)}")

        todo_ids = [
            lid for lid in leaf_ids
            if id2label[lid] in doc_labels
            and not (args.resume and id2label[lid] in result[leaf_layer_key])
        ]
        print(f"Leaf nodes to process: {len(todo_ids)}")

        save_lock  = asyncio.Lock()
        done_count = 0

        async def process_and_save(lid):
            nonlocal done_count
            label = id2label[lid]
            desc  = await process_leaf(client, args.model, label, docs_df, args.sample_n)
            async with save_lock:
                done_count += 1
                if desc is not None:
                    result[leaf_layer_key][label] = desc
                if done_count % 10 == 0 or done_count == len(todo_ids):
                    with open(out_path, 'w') as f:
                        json.dump({args.dataset: _clean_for_save(result)}, f, indent=4, ensure_ascii=False)
                    print(f"  Saved ({done_count}/{len(todo_ids)})")

        sem  = asyncio.Semaphore(args.concurrency)
        pbar = tqdm(total=len(todo_ids), desc="Leaf nodes")

        async def _run(lid):
            async with sem:
                await process_and_save(lid)
            pbar.update(1)

        await asyncio.gather(*[_run(lid) for lid in todo_ids])
        pbar.close()
        with open(out_path, 'w') as f:
            json.dump({args.dataset: _clean_for_save(result)}, f, indent=4, ensure_ascii=False)
        print(f"Leaf descriptions done: {len(result[leaf_layer_key])} nodes.")

    # ── Step 2: Parent descriptions (bottom-up) ───────────────────────────────
    if not args.discriminative_only:
        print("\n=== Generating parent descriptions ===")
        for layer_idx in range(leaf_layer - 1, -1, -1):
            layer_key       = f"layer {layer_idx}"
            child_layer_key = f"layer {layer_idx + 1}"

            todo_parents = [
                pid for pid in layer2ids[layer_idx]
                if not ((args.resume or args.parents_only) and id2label[pid] in result[layer_key])
            ]
            print(f"Layer {layer_idx}: {len(todo_parents)} to process")

            # Capture per-iteration values to avoid closure issues
            _layer_key       = layer_key
            _child_layer_key = child_layer_key
            _sem             = asyncio.Semaphore(args.concurrency)
            _save_lock       = asyncio.Lock()
            _done            = [0]

            async def _process_parent(pid, lk=_layer_key, clk=_child_layer_key, sem=_sem,
                                      save_lock=_save_lock, done=_done, total=len(todo_parents)):
                async with sem:
                    plabel = id2label[pid]
                    children_descs = {
                        id2label[cid]: result.get(clk, {}).get(id2label[cid])
                        for cid in id2info[pid].get('children', [])
                        if result.get(clk, {}).get(id2label[cid])
                    }
                    if not children_descs:
                        print(f"  WARNING: No child descriptions for '{plabel}', skipping.")
                        return
                    desc = await process_parent(client, args.model, plabel, children_descs)
                    async with save_lock:
                        done[0] += 1
                        if desc is not None:
                            result[lk][plabel] = desc
                        if done[0] % 5 == 0 or done[0] == total:
                            with open(out_path, 'w') as f:
                                json.dump({args.dataset: _clean_for_save(result)}, f,
                                          indent=4, ensure_ascii=False)

            pbar = tqdm(total=len(todo_parents), desc=f"Layer {layer_idx}")

            async def _wrap_parent(p):
                await _process_parent(p)
                pbar.update(1)

            await asyncio.gather(*[_wrap_parent(pid) for pid in todo_parents])
            pbar.close()

            with open(out_path, 'w') as f:
                json.dump({args.dataset: _clean_for_save(result)}, f, indent=4, ensure_ascii=False)
            print(f"  Saved layer {layer_idx}: {len(result[layer_key])} parent descriptions")

    # ── Step 3: Discriminative descriptions ───────────────────────────────────
    print("\n=== Generating discriminative descriptions ===")
    for layer_idx in range(num_layers):
        layer_key   = f"layer {layer_idx}"
        layer_descs = result.get(layer_key, {})
        if len(layer_descs) < 2:
            continue

        print(f"\n  Layer {layer_idx}: computing embeddings for {len(layer_descs)} classes...")
        similar_map = find_similar_classes(layer_descs, threshold=args.sim_threshold, drop_ratio=args.sim_drop_ratio)
        has_similar = {k: v for k, v in similar_map.items() if v}
        print(f"  {len(has_similar)}/{len(layer_descs)} classes have confusable neighbours")

        todo_labels = [
            label for label, neighbors in has_similar.items()
            if not (args.resume and 'detail_description' in layer_descs.get(label, {}))
        ]
        if not todo_labels:
            print(f"  Layer {layer_idx}: all done, skipping")
            continue

        disc_sem   = asyncio.Semaphore(args.concurrency)
        disc_lock  = asyncio.Lock()
        disc_count = [0]

        async def process_discriminative(label, lk=layer_key, ld=layer_descs,
                                         sm=similar_map, sem=disc_sem,
                                         lock=disc_lock, count=disc_count,
                                         total=len(todo_labels)):
            async with sem:
                neighbors     = sm[label]
                target_desc   = ld[label]
                similar_descs = [(sl, ld.get(sl, {}), ss) for sl, ss in neighbors]
                disc_result   = await generate_discriminative(client, args.model, label, target_desc, similar_descs)
                async with lock:
                    count[0] += 1
                    if disc_result is not None:
                        if 'detail_description' in disc_result:
                            result[lk][label]['detail_description'] = disc_result['detail_description']
                        result[lk][label]['similar_classes'] = [
                            {'label': sl, 'similarity': ss} for sl, ss in neighbors
                        ]
                    if count[0] % 10 == 0 or count[0] == total:
                        with open(out_path, 'w') as f:
                            json.dump({args.dataset: _clean_for_save(result)}, f,
                                      indent=4, ensure_ascii=False)

        pbar = tqdm(total=len(todo_labels), desc=f"Discriminative layer {layer_idx}")

        async def _wrap_disc(l):
            await process_discriminative(l)
            pbar.update(1)

        await asyncio.gather(*[_wrap_disc(label) for label in todo_labels])
        pbar.close()

        with open(out_path, 'w') as f:
            json.dump({args.dataset: _clean_for_save(result)}, f, indent=4, ensure_ascii=False)
        print(f"  Saved layer {layer_idx} discriminative descriptions")

    print(f"\nDone!")
    for layer_idx in range(num_layers):
        layer_key  = f"layer {layer_idx}"
        total      = len(result.get(layer_key, {}))
        with_disc  = sum(1 for v in result.get(layer_key, {}).values()
                         if isinstance(v, dict) and 'detail_description' in v)
        print(f"  Layer {layer_idx}: {total} nodes ({with_disc} with discriminative)")
    print(f"Output: {out_path}")


if __name__ == '__main__':
    asyncio.run(main())

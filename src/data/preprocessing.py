"""
LSD600 Preprocessing and BRAT Standoff Utilities.

Handles parsing of BRAT annotations (.ann and .txt), extraction of Lifestyle Factor
and Disease entities, multi-label relation identification, and entity masking
using special tokens for transformer-based Relation Extraction.
"""

import os
import re
from typing import Dict, List, Tuple, Any, Optional, Set
import pandas as pd

# The 8 relation classes in LSD600 taxonomy
RELATION_CLASSES = [
    "causes",
    "controls",
    "prevents",
    "treats",
    "statistical_association",
    "positive_statistical_association",
    "negative_statistical_association",
    "no_statistical_association",
]

RELATION2ID = {rel: idx for idx, rel in enumerate(RELATION_CLASSES)}
ID2RELATION = {idx: rel for idx, rel in enumerate(RELATION_CLASSES)}



def parse_brat_file(ann_path: str, txt_path: str) -> Dict[str, Any]:
    """
    Parses a single BRAT document: (.ann) annotations and (.txt) raw text.

    Returns:
        dict with keys:
            - 'text': raw abstract text
            - 'entities': dict of entity_id -> {id, type, spans: [(start, end)], text}
            - 'relations': list of {id, type, arg1, arg2}
            - 'equivalences': list of sets of equivalent entity_ids
    """
    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()

    entities = {}
    relations = []
    equivalences = []

    if not os.path.exists(ann_path):
        return {
            "text": text,
            "entities": entities,
            "relations": relations,
            "equivalences": equivalences,
        }

    with open(ann_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Entities start with 'T'
            if line.startswith("T"):
                parts = line.split("\t")
                if len(parts) >= 3:
                    tid = parts[0]
                    type_spans = parts[1].split()
                    etype = type_spans[0]
                    # Handle discontinuous or single spans: e.g., "18 24" or "18 24;30 35"
                    span_parts = " ".join(type_spans[1:]).split(";")
                    spans = []
                    for sp in span_parts:
                        s_e = sp.split()
                        if len(s_e) == 2:
                            spans.append((int(s_e[0]), int(s_e[1])))
                    ent_text = parts[2]
                    entities[tid] = {
                        "id": tid,
                        "type": etype,
                        "spans": spans,
                        "text": ent_text,
                    }

            # Relations start with 'R'
            elif line.startswith("R"):
                parts = line.split("\t")
                if len(parts) >= 2:
                    rid = parts[0]
                    rel_info = parts[1].split()
                    rtype = rel_info[0]
                    args = {}
                    for item in rel_info[1:]:
                        if ":" in item:
                            k, v = item.split(":", 1)
                            args[k] = v
                    relations.append({
                        "id": rid,
                        "type": rtype,
                        "arg1": args.get("Arg1", ""),
                        "arg2": args.get("Arg2", ""),
                    })

            # Equivalence relations start with '*'
            elif line.startswith("*"):
                parts = line.split("\t")
                if len(parts) >= 2 and parts[1].startswith("Equiv"):
                    equiv_items = set(parts[1].split()[1:])
                    equivalences.append(equiv_items)

    return {
        "text": text,
        "entities": entities,
        "relations": relations,
        "equivalences": equivalences,
    }


def mask_entities_for_re(
    text: str,
    focus_span_1: Tuple[int, int],
    focus_span_2: Tuple[int, int],
    other_spans: Optional[List[Tuple[int, int]]] = None,
    focus_token: str = "[unused1]",
    other_token: str = "[unused2]",
) -> str:
    """
    Masks entity spans in the text.
    The two candidate focus entities are replaced by `focus_token` (e.g. [unused1]),
    and other entities are replaced by `other_token` (e.g. [unused2]),
    matching the base paper's entity masking protocol.
    """
    replacements = []

    # Focus entities
    replacements.append((focus_span_1[0], focus_span_1[1], focus_token))
    replacements.append((focus_span_2[0], focus_span_2[1], focus_token))

    # Other entities
    if other_spans:
        for span in other_spans:
            # Avoid masking if overlapping with focus spans
            if (span[0] >= focus_span_1[1] or span[1] <= focus_span_1[0]) and \
               (span[0] >= focus_span_2[1] or span[1] <= focus_span_2[0]):
                replacements.append((span[0], span[1], other_token))

    # Sort replacements in reverse order of start offset to replace from end to beginning
    replacements.sort(key=lambda x: x[0], reverse=True)

    result_text = text
    for start, end, token in replacements:
        if 0 <= start <= end <= len(result_text):
            result_text = result_text[:start] + token + result_text[end:]

    return result_text


def extract_candidate_pairs_from_doc(doc: Dict[str, Any], doc_id: str = "") -> List[Dict[str, Any]]:
    """
    Extracts all candidate (Lifestyle Factor, Disease) entity pairs from a parsed BRAT document.
    Since relations are undirected, each pair is evaluated for active relation types.
    """
    text = doc["text"]
    entities = doc["entities"]
    relations = doc["relations"]

    # Filter into LSF and Disease entities
    lsf_entities = [e for e in entities.values() if e["type"] == "lifestyle_factor"]
    dis_entities = [e for e in entities.values() if e["type"] == "disease"]

    # Build a lookup for active relations between entity pairs
    pair_to_relations: Dict[Tuple[str, str], Set[str]] = {}
    for rel in relations:
        rtype = rel["type"]
        if rtype not in RELATION2ID:
            continue
        a1, a2 = rel["arg1"], rel["arg2"]
        pair_to_relations.setdefault((a1, a2), set()).add(rtype)
        pair_to_relations.setdefault((a2, a1), set()).add(rtype)

    # All spans of all valid entities for other-entity masking
    all_spans = [e["spans"][0] for e in entities.values() if e["spans"] and e["type"] in ("lifestyle_factor", "disease")]

    candidates = []
    for lsf in lsf_entities:
        for dis in dis_entities:
            lsf_id = lsf["id"]
            dis_id = dis["id"]
            if not lsf["spans"] or not dis["spans"]:
                continue

            lsf_span = lsf["spans"][0]
            dis_span = dis["spans"][0]

            active_rels = pair_to_relations.get((lsf_id, dis_id), set())

            # Multi-hot label vector for the 8 classes
            labels = [1 if rel in active_rels else 0 for rel in RELATION_CLASSES]

            masked_text = mask_entities_for_re(
                text=text,
                focus_span_1=lsf_span,
                focus_span_2=dis_span,
                other_spans=all_spans,
            )

            candidates.append({
                "doc_id": doc_id,
                "lsf_id": lsf_id,
                "dis_id": dis_id,
                "lsf_text": lsf["text"],
                "dis_text": dis["text"],
                "lsf_span": lsf_span,
                "dis_span": dis_span,
                "active_relations": list(active_rels),
                "has_relation": int(len(active_rels) > 0),
                "labels": labels,
                "masked_text": masked_text,
                "raw_text": text,
            })

    return candidates


def load_brat_dataset(dataset_dir: str) -> List[Dict[str, Any]]:
    """
    Loads all BRAT documents (.ann and .txt) in a directory and extracts candidate pairs.
    """
    files = os.listdir(dataset_dir)
    txt_files = [f for f in files if f.endswith(".txt")]
    all_candidates = []

    for txt_file in sorted(txt_files):
        doc_id = os.path.splitext(txt_file)[0]
        txt_path = os.path.join(dataset_dir, txt_file)
        ann_path = os.path.join(dataset_dir, f"{doc_id}.ann")

        doc = parse_brat_file(ann_path, txt_path)
        candidates = extract_candidate_pairs_from_doc(doc, doc_id=doc_id)
        all_candidates.extend(candidates)

    return all_candidates

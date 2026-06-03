"""
run_pipeline.py — End-to-end runner for Phase 1 (CSV-based evaluation).

Runs one or more methods on the test split of train_annotations.csv,
evaluates with both mIoU and the ranking table, and prints a comparison.

Usage examples
--------------
# Build Method 1 dictionary and evaluate:
python -m Link.run_pipeline --method 1 \
    --annotations train_annotations.csv \
    --notes train_notes.csv

# Evaluate all 4 methods (requires SNOMED RF2 + Ollama running locally):
python -m Link.run_pipeline --method all \
    --annotations train_annotations.csv \
    --notes train_notes.csv \
    --snomed-rf2 /path/to/sct2_Description_Snapshot_INT_*.txt

# Skip rebuilding if artifacts already exist:
python -m Link.run_pipeline --method 1 --no-rebuild \
    --annotations train_annotations.csv \
    --notes train_notes.csv
"""

import argparse
import csv
import gc
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from Link.evaluator import (
    evaluate,
    load_gold_annotations,
    print_comparison_table,
    print_results,
)


def load_notes(notes_csv: str) -> Dict[str, str]:
    notes = {}
    with open(notes_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            notes[row["note_id"]] = row["text"]
    return notes


def get_test_note_ids(annotations_csv: str) -> List[str]:
    ids = set()
    with open(annotations_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("annotation_type", "").lower() == "test":
                ids.add(row["note_id"])
    return sorted(ids)


def run_method1(
    notes: Dict[str, str],
    test_note_ids: List[str],
    annotations_csv: str,
    snomed_rf2: str,
    dict_path: str,
    rebuild: bool,
    snomed_rf2_fr: str = None,
) -> List[dict]:
    from Link.method1_dictionary import (
        build_dictionary,
        load_dictionary,
        predict,
        save_dictionary,
    )

    if rebuild or not Path(dict_path).exists():
        print("\n[Method 1] Building dictionary ...")
        d = build_dictionary(annotations_csv, snomed_rf2, snomed_rf2_fr)
        save_dictionary(d, dict_path)
    else:
        print(f"\n[Method 1] Loading existing dictionary from {dict_path}")

    d = load_dictionary(dict_path)

    predictions = []
    total = len(test_note_ids)
    print(f"\n[Method 1] Predicting on {total} test notes ...")
    for i, note_id in enumerate(test_note_ids, 1):
        if note_id not in notes:
            continue
        preds = predict(notes[note_id], d)
        for p in preds:
            p["note_id"] = note_id
        predictions.extend(preds)
        bar_filled = int(30 * i / total)
        bar = "#" * bar_filled + "-" * (30 - bar_filled)
        print(f"  [{bar}] {i}/{total} notes  ({len(predictions)} predictions so far)", end="\r")

    print(f"  [{'#' * 30}] {total}/{total} notes  ({len(predictions)} predictions total)    ")
    return predictions


def run_method2(
    notes: Dict[str, str],
    test_note_ids: List[str],
    snomed_rf2: str,
    index_dir: str,
    rebuild: bool,
    dict_path: str = "",
    annotations_csv: str = "",
) -> List[dict]:
    from Link.method2_sapbert import SapBERTIndex, build_index, predict

    if rebuild or not Path(index_dir, "faiss_index.bin").exists():
        if not snomed_rf2:
            print("[Method 2] Skipped: --snomed-rf2 required to build index")
            return []
        print("\n[Method 2] Building SapBERT FAISS index ...")
        build_index(snomed_rf2, index_dir)

    if not Path(index_dir, "faiss_index.bin").exists():
        print("[Method 2] Skipped: index not found")
        return []

    dictionary = None
    if dict_path:
        from Link.method1_dictionary import build_dictionary, load_dictionary, save_dictionary
        if not Path(dict_path).exists():
            if annotations_csv:
                print(f"\n[Method 2] Building Method 1 dictionary for span pre-filtering ...")
                d = build_dictionary(annotations_csv, snomed_rf2)
                save_dictionary(d, dict_path)
            else:
                print("[Method 2] No dictionary found and no --annotations provided — using sliding window")
        if Path(dict_path).exists():
            print(f"[Method 2] Loading dictionary for span pre-filtering from {dict_path} ...")
            dictionary = load_dictionary(dict_path)

    idx = SapBERTIndex(index_dir)
    predictions = []
    total = len(test_note_ids)
    for i, note_id in enumerate(test_note_ids, 1):
        if note_id not in notes:
            continue
        preds = predict(notes[note_id], idx, dictionary=dictionary)
        for p in preds:
            p["note_id"] = note_id
        predictions.extend(preds)
        bar_filled = int(30 * i / total)
        bar = "#" * bar_filled + "-" * (30 - bar_filled)
        print(f"  [{bar}] {i}/{total} notes  ({len(predictions)} predictions so far)", end="\r")

    print(f"  [{'#' * 30}] {total}/{total} notes  ({len(predictions)} predictions total)    ")
    return predictions


def run_method3(
    notes: Dict[str, str],
    test_note_ids: List[str],
    snomed_rf2: str,
    index_dir: str,
    rebuild: bool,
    llm_model: str = "llama3.1:8b",
) -> List[dict]:
    from Link.method3_llm_rag import OllamaClient, RAGIndex, build_index, predict

    if rebuild or not Path(index_dir, "faiss_index.bin").exists():
        if not snomed_rf2:
            print("[Method 3] Skipped: --snomed-rf2 required to build index")
            return []
        print("\n[Method 3] Building RAG FAISS index ...")
        build_index(snomed_rf2, index_dir)

    if not Path(index_dir, "faiss_index.bin").exists():
        print("[Method 3] Skipped: index not found")
        return []

    idx = RAGIndex(index_dir)
    client = OllamaClient(model_name=llm_model)
    print(f"  LLM: {llm_model} (local Ollama)")

    predictions = []
    total = len(test_note_ids)
    print(f"\n[Method 3] Predicting on {total} test notes (NER + rerank) ...")
    for i, note_id in enumerate(test_note_ids, 1):
        if note_id not in notes:
            continue
        preds = predict(notes[note_id], idx, client, use_llm_reranking=True)
        for p in preds:
            p["note_id"] = note_id
        predictions.extend(preds)
        gc.collect()
        bar_filled = int(30 * i / total)
        bar = "#" * bar_filled + "-" * (30 - bar_filled)
        print(f"  [{bar}] {i}/{total} notes  ({len(predictions)} predictions so far)", end="\r")

    print(f"  [{'#' * 30}] {total}/{total} notes  ({len(predictions)} predictions total)    ")
    return predictions


def run_method4(
    notes: Dict[str, str],
    test_note_ids: List[str],
    snomed_rf2: str,
    rag_index_dir: str,
    dict_path: str,
    annotations_csv: str,
    rebuild: bool,
    llm_model: str = "llama3.1:8b",
    use_llm_reranking: bool = True,
    snomed_rf2_fr: str = None,
) -> List[dict]:
    from Link.method4_hybrid import predict
    from Link.method3_llm_rag import RAGIndex, OllamaClient
    from Link.method1_dictionary import build_dictionary, load_dictionary, save_dictionary

    if rebuild or not Path(dict_path).exists():
        if not annotations_csv:
            print("[Method 4] Skipped: --annotations required to build dictionary")
            return []
        print("\n[Method 4] Building Method 1 dictionary ...")
        d = build_dictionary(annotations_csv, snomed_rf2, snomed_rf2_fr)
        save_dictionary(d, dict_path)
    dictionary = load_dictionary(dict_path)

    if not Path(rag_index_dir, "faiss_index.bin").exists():
        if not snomed_rf2:
            print("[Method 4] Skipped: --snomed-rf2 required to build RAG index")
            return []
        from Link.method3_llm_rag import build_index
        print("\n[Method 4] Building RAG FAISS index ...")
        build_index(snomed_rf2, rag_index_dir)

    if not Path(rag_index_dir, "faiss_index.bin").exists():
        print("[Method 4] Skipped: RAG index not found")
        return []

    rag_index = RAGIndex(rag_index_dir)
    client = OllamaClient(model_name=llm_model)
    mode = "LLM NER + tiered dict + FAISS + reranking" if use_llm_reranking else "LLM NER + tiered dict + FAISS"
    print(f"  LLM: {llm_model} ({mode})")

    predictions = []
    total = len(test_note_ids)
    print(f"\n[Method 4] Predicting on {total} test notes ({mode}) ...")
    for i, note_id in enumerate(test_note_ids, 1):
        if note_id not in notes:
            continue
        preds = predict(notes[note_id], dictionary, rag_index, client, use_llm_reranking=use_llm_reranking)
        for p in preds:
            p["note_id"] = note_id
        predictions.extend(preds)
        bar_filled = int(30 * i / total)
        bar = "#" * bar_filled + "-" * (30 - bar_filled)
        print(f"  [{bar}] {i}/{total} notes  ({len(predictions)} predictions so far)", end="\r")

    print(f"  [{'#' * 30}] {total}/{total} notes  ({len(predictions)} predictions total)    ")
    return predictions


def main():
    parser = argparse.ArgumentParser(description="Phase 1 pipeline runner")
    parser.add_argument(
        "--method", default="1", choices=["1", "2", "3", "4", "all"],
        help="Which method(s) to run"
    )
    parser.add_argument("--annotations", default="train_annotations.csv")
    parser.add_argument("--notes",       default="train_notes.csv")
    parser.add_argument("--snomed-rf2",  default=None,
                        help="SNOMED CT RF2 English description file (optional for Method 1)")
    parser.add_argument("--snomed-rf2-fr", default=None,
                        help="SNOMED CT RF2 French description file")
    parser.add_argument("--results-dir", default=str(Path(__file__).parent / "results"),
                        help="Directory for artifacts (dictionaries, indexes)")
    parser.add_argument("--no-rebuild",  dest="rebuild", action="store_false", default=True,
                        help="Re-use existing build artifacts instead of rebuilding")
    parser.add_argument("--llm-model",   default="llama3.1:8b",
                        help="Methods 3 & 4: Ollama model name (e.g. llama3.1:8b)")
    parser.add_argument("--max-notes",   type=int, default=None,
                        help="Limit number of test notes processed (for quick smoke tests)")
    parser.add_argument("--no-rerank",   dest="rerank", action="store_false", default=True,
                        help="Method 4: skip LLM reranking (fast FAISS-only mode)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("Loading notes and gold annotations ...")
    notes = load_notes(args.notes)
    gold = load_gold_annotations(args.annotations, split="test")
    test_ids = get_test_note_ids(args.annotations)
    if args.max_notes:
        test_ids = test_ids[:args.max_notes]
    print(f"  {len(notes)} notes total, {len(test_ids)} test notes, {len(gold)} gold annotations")

    methods_to_run = ["1", "2", "3", "4"] if args.method == "all" else [args.method]
    all_results = []

    if "1" in methods_to_run:
        preds = run_method1(
            notes, test_ids, args.annotations, args.snomed_rf2,
            dict_path=str(results_dir / "dictionary.json"),
            rebuild=args.rebuild,
            snomed_rf2_fr=args.snomed_rf2_fr,
        )
        if preds:
            res = evaluate(gold, preds)
            print_results(res, method_name="Method 1 — Dictionary + Fuzzy")
            all_results.append(("Method 1", res))

    if "2" in methods_to_run:
        preds = run_method2(
            notes, test_ids, args.snomed_rf2,
            index_dir=str(results_dir / "sapbert_index"),
            rebuild=args.rebuild,
            dict_path=str(results_dir / "dictionary.json"),
            annotations_csv=args.annotations,
        )
        if preds:
            res = evaluate(gold, preds)
            print_results(res, method_name="Method 2 — SapBERT")
            all_results.append(("Method 2", res))

    if "3" in methods_to_run:
        preds = run_method3(
            notes, test_ids, args.snomed_rf2,
            index_dir=str(results_dir / "rag_index"),
            rebuild=args.rebuild,
            llm_model=args.llm_model,
        )
        if preds:
            res = evaluate(gold, preds)
            print_results(res, method_name="Method 3 — LLM + RAG")
            all_results.append(("Method 3", res))

    if "4" in methods_to_run:
        preds = run_method4(
            notes, test_ids, args.snomed_rf2,
            rag_index_dir=str(results_dir / "rag_index"),
            dict_path=str(results_dir / "dictionary.json"),
            annotations_csv=args.annotations,
            rebuild=args.rebuild,
            llm_model=args.llm_model,
            use_llm_reranking=args.rerank,
            snomed_rf2_fr=args.snomed_rf2_fr,
        )
        if preds:
            res = evaluate(gold, preds)
            print_results(res, method_name="Method 4 — Hybrid")
            all_results.append(("Method 4", res))

    if len(all_results) > 1:
        print("\nComparison across all methods:")
        print_comparison_table(all_results)


if __name__ == "__main__":
    main()

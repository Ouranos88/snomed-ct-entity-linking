"""
Two evaluation metrics for SNOMED CT entity linking:
  1. mIoU — strict end-to-end (span IoU × concept_correct)
  2. Ranking table — MRR / NDCG / Linear / Hits@k with IoU ≥ 0.3 gate
"""

import csv
import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


def _iou(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    overlap = max(0, min(a_end, b_end) - max(a_start, b_start))
    if overlap == 0:
        return 0.0
    union = (a_end - a_start) + (b_end - b_start) - overlap
    return overlap / union if union > 0 else 0.0


def _best_pred(gold: dict, preds: list) -> Tuple[float, Optional[dict]]:
    best_iou, best = 0.0, None
    for p in preds:
        iou = _iou(int(gold["start"]), int(gold["end"]), p["start"], p["end"])
        if iou > best_iou:
            best_iou, best = iou, p
    return best_iou, best


def _rank_of(candidates: list, gold_id: str) -> Optional[int]:
    for i, c in enumerate(candidates):
        if str(c["concept_id"]) == str(gold_id):
            return i + 1
    return None


def _dcg(rank: int) -> float:
    return 1.0 / math.log2(rank + 1)


def _linear(rank: int, k: int = 5) -> float:
    return max(0.0, (k - rank + 1) / k) if rank <= k else 0.0


def _evaluate_note(gold_list: list, pred_list: list, iou_gate: float = 0.3, k: int = 5) -> dict:
    # mIoU
    iou_scores = []
    for gold in gold_list:
        best_iou, best = _best_pred(gold, pred_list)
        if best is None or best_iou == 0.0:
            iou_scores.append(0.0)
        else:
            correct = 1.0 if str(best["candidates"][0]["concept_id"]) == str(gold["concept_id"]) else 0.0
            iou_scores.append(best_iou * correct)

    miou = sum(iou_scores) / len(iou_scores) if iou_scores else 0.0

    # Span F1
    tp_gold = sum(1 for g in gold_list if _best_pred(g, pred_list)[0] >= 0.5)
    tp_pred = sum(
        1 for p in pred_list
        if any(_iou(int(g["start"]), int(g["end"]), p["start"], p["end"]) >= 0.5 for g in gold_list)
    )
    prec = tp_pred / len(pred_list) if pred_list else 0.0
    rec  = tp_gold / len(gold_list) if gold_list else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    # Ranking metrics
    mrr_s, ndcg_s, lin_s = [], [], []
    hits = {1: 0, 3: 0, 5: 0}
    gated = 0

    for gold in gold_list:
        best_iou, best = _best_pred(gold, pred_list)
        if best is None or best_iou < iou_gate:
            continue
        gated += 1
        rank = _rank_of(best["candidates"], gold["concept_id"])
        if rank is None:
            mrr_s.append(0.0); ndcg_s.append(0.0); lin_s.append(0.0)
        else:
            mrr_s.append(1.0 / rank)
            ndcg_s.append(_dcg(rank))   # ideal DCG = _dcg(1) = 1.0, so no division needed
            lin_s.append(_linear(rank, k))
            for h in [1, 3, 5]:
                if rank <= h:
                    hits[h] += 1

    n = gated if gated > 0 else 1
    return {
        "n_gold": len(gold_list), "n_pred": len(pred_list),
        "miou": miou,
        "span_precision": prec, "span_recall": rec, "span_f1": f1,
        "gated_count": gated,
        "mrr":    sum(mrr_s)  / n if mrr_s  else 0.0,
        "ndcg":   sum(ndcg_s) / n if ndcg_s else 0.0,
        "linear": sum(lin_s)  / n if lin_s  else 0.0,
        "hits1": hits[1] / n, "hits3": hits[3] / n, "hits5": hits[5] / n,
    }


def evaluate(gold_annotations: list, predictions: list,
             iou_gate: float = 0.3, k: int = 5) -> dict:
    gold_by_note: Dict[str, list] = defaultdict(list)
    pred_by_note: Dict[str, list] = defaultdict(list)
    for g in gold_annotations:
        gold_by_note[str(g["note_id"])].append(g)
    for p in predictions:
        pred_by_note[str(p["note_id"])].append(p)

    note_results = [
        {**_evaluate_note(gold_by_note[nid], pred_by_note.get(nid, []), iou_gate, k), "note_id": nid}
        for nid in gold_by_note
    ]

    def macro(key: str) -> float:
        vals = [r[key] for r in note_results]
        return sum(vals) / len(vals) if vals else 0.0

    return {
        "note_results":             note_results,
        "n_notes":                  len(note_results),
        "total_gold_annotations":   sum(r["n_gold"] for r in note_results),
        "total_predictions":        sum(r["n_pred"] for r in note_results),
        "total_gated_annotations":  sum(r["gated_count"] for r in note_results),
        "miou":           macro("miou"),
        "span_precision": macro("span_precision"),
        "span_recall":    macro("span_recall"),
        "span_f1":        macro("span_f1"),
        "mrr":            macro("mrr"),
        "ndcg":           macro("ndcg"),
        "linear":         macro("linear"),
        "hits1":          macro("hits1"),
        "hits3":          macro("hits3"),
        "hits5":          macro("hits5"),
    }


def print_results(results: dict, method_name: str = "Method") -> None:
    print(f"\n{'='*60}")
    print(f"  {method_name}")
    print(f"{'='*60}")
    print(f"  Notes evaluated         : {results['n_notes']}")
    print(f"  Total gold annotations  : {results['total_gold_annotations']}")
    print(f"  Total predictions       : {results['total_predictions']}")
    print()
    print(f"  --- mIoU (strict end-to-end) ---")
    print(f"  mIoU            : {results['miou']:.4f}")
    print(f"  Span Precision  : {results['span_precision']:.4f}")
    print(f"  Span Recall     : {results['span_recall']:.4f}")
    print(f"  Span F1         : {results['span_f1']:.4f}")
    print()
    print(f"  --- Ranking (IoU ≥ 0.3 gate) ---")
    print(f"  Gated annotations : {results['total_gated_annotations']}")
    print(f"  MRR        : {results['mrr']:.4f}")
    print(f"  NDCG@5     : {results['ndcg']:.4f}")
    print(f"  Linear@5   : {results['linear']:.4f}")
    print(f"  Hits@1     : {results['hits1']:.4f}")
    print(f"  Hits@3     : {results['hits3']:.4f}")
    print(f"  Hits@5     : {results['hits5']:.4f}")
    print(f"{'='*60}\n")


def print_comparison_table(results_list: List[Tuple[str, dict]]) -> None:
    metrics = [
        ("mIoU",     "miou"),
        ("Span F1",  "span_f1"),
        ("MRR",      "mrr"),
        ("NDCG@5",   "ndcg"),
        ("Linear@5", "linear"),
        ("Hits@1",   "hits1"),
        ("Hits@3",   "hits3"),
        ("Hits@5",   "hits5"),
    ]
    col_w = 12
    header = f"{'Metric':<18}" + "".join(f"{name:>{col_w}}" for name, _ in results_list)
    sep = "=" * (18 + col_w * len(results_list))
    print("\n" + sep)
    print(header)
    print("-" * (18 + col_w * len(results_list)))
    for label, key in metrics:
        print(f"{label:<18}" + "".join(f"{res[key]:>{col_w}.4f}" for _, res in results_list))
    print(sep + "\n")


def load_gold_annotations(annotations_csv: str, split: str = "test") -> list:
    gold = []
    with open(annotations_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if split != "all" and row.get("annotation_type", "").lower() != split:
                continue
            gold.append({
                "note_id":    row["note_id"],
                "start":      int(float(row["start"])),
                "end":        int(float(row["end"])),
                "span":       row["span"],
                "concept_id": row["concept_id"],
            })
    return gold

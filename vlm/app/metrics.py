import pandas as pd
from evaluate import load
from rouge_score import rouge_scorer

def rouge(predictions: list[str], references: list[str], r_type: str = "", results_table: pd.DataFrame | None = None):
    """
    Computes the ROUGE score for the given predictions and references.

    Args:
        predictions (list[str]): List of predicted strings.
        references (list[str]): List of reference strings.
        r_type (str): Type of ROUGE score to compute (e.g., "rouge1", "rougeL").
        results_table (pd.DataFrame | None): DataFrame to store the computed scores.

    Returns:
        tuple: Tuple containing the F1 score, precision, recall, and the results_table DataFrame (if provided else `None`).
    """
    precision = []
    recall = []
    f1 = []

    scorer = rouge_scorer.RougeScorer([r_type], use_stemmer=True)
    for ref, pred in zip(references, predictions):
        score = scorer.score(ref, pred)
        precision.append(score[r_type].precision)
        recall.append(score[r_type].recall)
        f1.append(score[r_type].fmeasure)

    if results_table is not None:
        results_table[f"{r_type}_precision"] = precision
        results_table[f"{r_type}_recall"] = recall
        results_table[f"{r_type}_fmeasure"] = f1

    f1 = sum(f1) / len(f1)
    precision = sum(precision) / len(precision)
    recall = sum(recall) / len(recall)
    return f1, precision, recall, results_table


def bertS(predictions: list[str], references: list[str], results_table: pd.DataFrame | None = None):
    """
    Compute the BERTScore for the given predictions and references.

    Args: 
        predictions (list[str]): List of predicted strings.
        references (list[str]): List of reference strings.
        results_table (pd.DataFrame | None): DataFrame to store the computed scores.

    Returns:
        tuple: Tuple containing the F1 score, precision, recall, and the results_table DataFrame (if provided else `None`).
    """
    bertscore = load("bertscore")
    results = bertscore.compute(predictions=predictions, references=references, lang="en")
    if not isinstance(results, dict):
        raise ValueError("BERTScore results should be a dictionary.")
    precision = results["precision"]
    recall = results["recall"]
    f1 = results["f1"]
    if results_table is not None:
        results_table["bertscore_precision"] = precision
        results_table["bertscore_recall"] = recall
        results_table["bertscore_f1"] = f1

    f1 = sum(results["f1"]) / len(results["f1"])
    precision = sum(results["precision"]) / len(results["precision"])
    recall = sum(results["recall"]) / len(results["recall"])
    return f1, precision, recall, results_table
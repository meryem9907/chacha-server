from os import makedirs, path
from pathlib import Path
from typing import Literal
from vlm.config import SCORES_PATH
import pandas as pd
from vlm.app.metrics import bertS, rouge

def compute_evaluation_scores(predictions:list, references:list, results_table: pd.DataFrame, dataset_name: Literal["scivqa", "hololens"], model_path:str):
    """
    Compute evaluation scores.
    The scores are computed using the ROUGE and BERTScore metrics.
    The scores are saved in the scores dir

    Args: 
        predictions (list): Predicted responses.
        references (list): References.
        results_table (pd.DataFrame): Table 
        dataset_name (pd.Dataframe): The name of the dataset to evaluate. 
    """

    if len(references) != len(predictions):
        raise ValueError("The lengths of references and predictions do not match.")

    # 1) Create or use the score path to save scores
    scores_path = path.join(SCORES_PATH)
    if not path.exists(scores_path):
        makedirs(scores_path)
    
    # 2) Get the metrics for rouge1, rougeL and bertscore then save as csv.
    rouge1_score_f1, rouge1_score_precision, rouge1_score_recall, results_table = rouge(
        predictions, references, "rouge1", results_table
    )
    rougeL_score_f1, rougeL_score_precision, rougeL_score_recall, results_table = rouge(
        predictions, references, "rougeL", results_table
    )
    bert_score_f1, bert_score_precision, bert_score_recall, results_table = bertS(predictions, references, results_table)
    results_table.to_csv(Path(SCORES_PATH) / f"{dataset_name}-results_final-{model_path}.csv", sep=";", index=False)

    # 3) Also write the mean results in a table
    metrics_df = pd.DataFrame(
        [
            {
                "Metric": "ROUGE-1",
                "F1 (%)": round(rouge1_score_f1 * 100, 3),
                "Precision (%)": round(rouge1_score_precision * 100, 3),
                "Recall (%)": round(rouge1_score_recall * 100, 3),
            },
            {
                "Metric": "ROUGE-L",
                "F1 (%)": round(rougeL_score_f1 * 100, 3),
                "Precision (%)": round(rougeL_score_precision * 100, 3),
                "Recall (%)": round(rougeL_score_recall * 100, 3),
            },
            {
                "Metric": "BERTScore",
                "F1 (%)": round(bert_score_f1 * 100, 3),
                "Precision (%)": round(bert_score_precision * 100, 3),
                "Recall (%)": round(bert_score_recall * 100, 3),
            },
        ]
    )
    print("\n%s", metrics_df.to_string(index=False))
    metrics_df.to_csv(path.join(SCORES_PATH, f"scores_{dataset_name}-{model_path}.csv"), sep=";", index=False)
   
    # 4) Create tables for metrics based on figure type and QA type. Save as csv.
    score_cols = [
    "rouge1_fmeasure","rouge1_precision","rouge1_recall",
    "rougeL_fmeasure","rougeL_precision","rougeL_recall",
    "bertscore_f1","bertscore_precision","bertscore_recall",
]

    metric_df = (
        results_table
        .groupby(["figure_type", "qa_pair_type"], as_index=False)[score_cols]
        .mean()
    )

    # round all score columns to 2 decimals
    metric_df[score_cols] = metric_df[score_cols].round(2)
    metric_df.to_csv(Path(SCORES_PATH) / f"{dataset_name}-filtered_metrics-{model_path}.csv", sep=";", index=False)

    
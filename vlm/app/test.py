from vlm.app.dataset_utils import merge_dataset_with_prompts_from_hololens, generate_hololens_dataset_from_sample_dataset, load_100_samples, load_saved_100_samples, filter_sampled_images
from vlm.config import DATA_PATH
from vlm.app.scoring import compute_evaluation_scores
from datasets import Image
from huggingface_hub import hf_hub_download
from pathlib import Path
import pandas as pd
from os import path

ds = pd.read_csv(path.join(DATA_PATH,"results.csv"), sep=";" )
preds = ds["prediction"].tolist() #[r["prediction"] for r in rows]
refs  = ds["gold"].tolist() #[r["gold"] for r in rows] 
compute_evaluation_scores(preds, refs, ds, "scivqa")
#dsN=load_saved_100_samples() 
#generate_hololens_dataset_from_sample_dataset(dsN)
# csv_path = Path(DATA_PATH) / "sample.csv"
#merge_csv_path = Path(DATA_PATH) / "merged_datasets.csv"
#merge_dataset_with_prompts_from_hololens(dsN).to_csv(merge_csv_path, sep=";",  index=False, encoding="utf-8-sig")
#
#pd = dsN.to_pandas()
#pd.to_csv(csv_path, sep=";", index=False,  encoding="utf-8-sig")
#print(len(dsN.to_list()))
#filter_sampled_images(dsN)


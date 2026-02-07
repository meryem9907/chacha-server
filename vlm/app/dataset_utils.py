from datasets import load_dataset, load_from_disk, Dataset, Value
import os, zipfile, shutil
from pathlib import Path
from huggingface_hub import hf_hub_download
from os import  path
import pandas as pd
import numpy as np
from vlm.config import IMAGES_PATH, DATA_PATH, DOWNLOADED_IMAGES_PATH, HOLOLENS_DATA_PATH

SEED = int(os.getenv("SCIVQA_SEED", "42"))
N = int(os.getenv("SCIVQA_N", "100"))
SPLIT = os.getenv("SCIVQA_SPLIT", "test")  

def load_n_samples(n):
    ds = load_dataset("katebor/SciVQA", split=SPLIT)
    dsN = ds.shuffle(seed=SEED).select(range(n))
    
    # save to data dir 
    Path(DATA_PATH).mkdir(parents=True, exist_ok=True)
    dsN.save_to_disk(DATA_PATH)
    return dsN

def get_stored_samples():
    dsN = load_from_disk(DATA_PATH)
    return dsN

# to get the images of the sample
def filter_sampled_images(dsN:Dataset):
    downloaded_img_path = Path(DOWNLOADED_IMAGES_PATH)
    Path(IMAGES_PATH).mkdir(parents=True, exist_ok=True)
    for d in dsN:
        print(downloaded_img_path)
        hits = list(downloaded_img_path.rglob(d["image_file"]))
        print(hits)
        if hits:
            shutil.copy2(Path(downloaded_img_path, d["image_file"]), Path(IMAGES_PATH))

def merge_dataset_with_prompts_from_hololens(control_dsN:Dataset):
    # merge on instance_id 
    hololens_q_data_path = path.join(HOLOLENS_DATA_PATH, "hololens_dataset_complete.csv")
    questions_df = pd.read_csv(hololens_q_data_path, sep=";")
    control_df = control_dsN.to_pandas()
    merged_df: pd.DataFrame = control_df.merge(questions_df[["hololens_question", "instance_id"]], on="instance_id", how="left", )
    merged_ds = Dataset.from_pandas(merged_df, preserve_index=False)
    return merged_ds
    
def generate_hololens_dataset_from_sample_dataset(dsN:Dataset):
    hololens_dsN = dsN.select_columns(["instance_id", "image_file", "question"])
    hololens_dsN=hololens_dsN.add_column(name="hololens_question", column=np.full(len(hololens_dsN.to_list()), "empty", dtype=object), feature=Value("string"))
    hololens_df = hololens_dsN.to_pandas()
    hololens_df.to_csv(Path(HOLOLENS_DATA_PATH) / "hololens_dataset_empty.csv", sep=";", index=False,  encoding="utf-8-sig")

    
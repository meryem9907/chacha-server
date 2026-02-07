import os, json, dotenv
from PIL import Image
from io import BytesIO
from vlm.app.dataset_utils import merge_dataset_with_prompts_from_hololens, generate_hololens_dataset_from_sample_dataset, get_stored_samples, load_n_samples, filter_sampled_images
from typing import Dict, Any, List,  Literal
from vlm.app.prompt_utils import build_dynamic_prompt
from pathlib import Path
from vlm.app.scoring import compute_evaluation_scores
from vlm.app.model import VisualLanguageModelForCharts
import pandas as pd
from vlm.config import IMAGES_PATH, HOLOLENS_IMAGES_PATH,  SCORES_PATH
from datasets import Dataset

def evaluate(vlm: VisualLanguageModelForCharts, eval_type:Literal["scivqa", "hololens"], model_path: str):
    # 0) load data
    dsN:Dataset = get_stored_samples()
    image_path = None
    model_path = model_path.replace("/", "-")

    # 1) Set the image pathes for the current evaluation.
    if (eval_type=="hololens"):
        print("Evaluating hololens dataset")
        image_path = HOLOLENS_IMAGES_PATH
    else:
        print("Evaluating scivqa dataset")
        image_path = IMAGES_PATH
    rows: List[Dict[str, Any]] = [] 
    print(f"Generating predictions with {eval_type} with images in: {image_path}")

    for data in dsN:
        # 2) Extract images
        print(f"Opening image file: {data.get("image_file")}")
        pillow_image = retrieve_image_file(images_dir=image_path, filename=data.get("image_file"))

        # 3) Reconstruct the prompt style from the paper "Instruction-tuned QwenChart for Chart Question Answering"
        dynamic_prompt = build_dynamic_prompt(entry=data)
        
        question = data.get("question")
        print(f"Prompt: {question}")
        gold = data["answer"]
        print(f"Gold: {gold}")

        pred = ""

        # 4) Generate a prediction with dynamic prompt as a system prompt
        try:
            pred = vlm.run_vlm(prompt=question, dynamic_prompt=dynamic_prompt, chart=pillow_image)
        except Exception as e:
            print(f"Error:{e}")
        print(f"Prediction: {pred}")

        # 5) Save prediction and gold answer as a row
        rows.append({
            "instance_id": data.get("instance_id"),
            "figure_id": data.get("figure_id"),
            "image_file": data.get("image_file"),
            "dynamic_prompt": dynamic_prompt,
            "answer_options": json.dumps(data.get("answer_options", []), ensure_ascii=False),
            "figure_type": data.get("figure_type"),
            "qa_pair_type": data.get("qa_pair_type"),
            "caption": data.get("caption"),
            "question": question,
            "gold": gold,
            "prediction": pred,
        })
    
    # 6) Create a dataframe from the rows and save as csv
    ds = pd.DataFrame(rows)
    ds.to_csv(Path(SCORES_PATH) / f"{eval_type}-results_tmp_{model_path}.csv", sep=";", index=False)
    print(f"Saved results in {Path(SCORES_PATH) / f"results_tmp_{model_path}.csv"}")

    preds = ds["prediction"].tolist() 
    refs  = ds["gold"].tolist() 
    print(f"Computing metrics")

    # 7) Measure the rouge and bertscore for each pred and also get the mean score from overall
    compute_evaluation_scores(predictions=preds, references=refs, results_table=ds, dataset_name=eval_type, model_path=model_path)

def retrieve_image_file(images_dir:str, filename:str):
    image_path = os.path.join(images_dir, filename)
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {image_path}")
    image_file = Image.open(BytesIO(path.read_bytes()))
    return image_file

def pre_test_setup():
    """
    Steps:
    - Run this function before generating the hololens dataset with the hololens glasses
    - Print the filtered images from the "/images" directory
    - Use the "hololens_data/hololens_dataset_empty.csv" to fill with questions spoken in 
    the VLM-VQA App on Hololens.
    - The chart pictures from hololens should mantain their original filename, which is the "image_file"
    - The chart pictures from hololens should be converted to .png (e.g. "mogrifiy" in linux: mogrify -format jpg *.png)
    - Save the hololens chart pictures in the directory: "hololens_data/images"
    """
    # 1) load a shuffled sample from scivqa dataset.
    # Also make sure to download all images "images_test.zip" from the 
    # dataset repo: 
    # https://huggingface.co/datasets/katebor/SciVQA/tree/main/data 
    # in the "/all_images" dir
    sample_dsN = load_n_samples(n=100)
    # 2) load the images from the sampled scivqa dataset into the images directory
    filter_sampled_images(sample_dsN)
    # 3) for the evaluation with hololens data create a dataset with an additional empty column for natural spoken questions
    generate_hololens_dataset_from_sample_dataset(sample_dsN)

def post_test_setup():
    sample_dsN = get_stored_samples()
    # 1) merge hololens questions into the sample dataset
    merge_dataset_with_prompts_from_hololens(sample_dsN)

if __name__== "__main__":
    ############### PREPARE EVALUATION DATA ###############
    # Execute these two one at a time before evaluation. Make sure the other functions are disabled/commented out 
    ############### BEFORE TESTING ON HOLOLENS ###############
    # pre_test_setup()
    ############### AFTER TESTING ON HOLOLENS ###############
    # post_test_setup()

    ############### EVALUATE ###############
    # Get config
    ENV_PATH = Path(__file__).resolve().parent.parent / ".env" 
    dotenv.load_dotenv(ENV_PATH)
    MODEL_NAME = os.getenv("MODEL_NAME") 
    FORCE_CPU = os.getenv("FORCE_CPU", "true").lower() == "true"
    
    # Load model
    print("Loading:", MODEL_NAME)
    vlm = VisualLanguageModelForCharts()
    vlm.load_model(MODEL_NAME, FORCE_CPU)

    # evaluate one at a time and store results in "/scores" directory
    # scivqa dataset
    #evaluate(vlm=vlm, eval_type="scivqa", model_path=MODEL_NAME)
    # hololens dataset
    evaluate(vlm=vlm, eval_type="hololens",  model_path=MODEL_NAME)
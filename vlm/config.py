from os import path
from pathlib import Path

BASE_PATH = Path(__file__).resolve().parent
DATA_PATH = path.join(BASE_PATH, "data")
SCORES_PATH = path.join(BASE_PATH, "scores")
IMAGES_PATH = path.join(BASE_PATH, "images")
DOWNLOADED_IMAGES_PATH = path.join(BASE_PATH, "all_images")
HOLOLENS_DATA_PATH = path.join(BASE_PATH, "hololens_data")
HOLOLENS_IMAGES_PATH = path.join(HOLOLENS_DATA_PATH, "images")

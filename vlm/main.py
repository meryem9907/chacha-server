import os, dotenv, base64, json
from PIL import Image
from io import BytesIO
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from app.model import VisualLanguageModelForCharts

#otenv.load_dotenv(".env")

MODEL_NAME = os.getenv("MODEL_NAME") 
print("model name is ", MODEL_NAME)
FORCE_CPU = os.getenv("FORCE_CPU", "true").lower() == "true"
MAX_NEW_TOKENS_DEFAULT = int(os.getenv("MAX_NEW_TOKENS", "128"))

class VLMRequest(BaseModel):
    """
    Request body schema for VLM inference.

    Args:
        query: Natural-language prompt/question to ask the model about the image.
        image_b64: Base64-encoded image bytes (no data URI prefix expected).
        extension: Image file extension hint (e.g., "png", "jpg"). Defaults to "png".
        max_new_tokens: Optional maximum number of tokens to generate. If None, adefault value is used.
    """
    query: str
    image_b64: str          
    extension: str = "png"  
    max_new_tokens: int | None = None

vlm = VisualLanguageModelForCharts()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler for startup/shutdown behavior.

    This context manager runs at application startup to load the VLM model and
    then yields control to allow the application to serve requests.

    Args:
        app: The FastAPI application instance.

    Yields:
        None
    """
    vlm.load_model(MODEL_NAME, FORCE_CPU)
    yield

app = FastAPI(lifespan=lifespan)

@app.post("/vlm/generate")
def generate(req: VLMRequest):
    """
    Generate a model response for a given prompt and base64-encoded image.

    This endpoint decodes the provided base64 image, loads it into a PIL Image,
    and runs the visual language model with the given prompt.

    Args:
        req: Request payload containing the prompt, base64 image, and optional
            generation parameters.

    Returns:
        A JSON object containing the generated text under the "text" key.

    Raises:
        HTTPException: If the base64 image is invalid (400) or model inference fails (500).
    """
    try:
        raw = base64.b64decode(req.image_b64)
        img = Image.open(BytesIO(raw))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image_b64: {e}")

    max_new_tokens = req.max_new_tokens or MAX_NEW_TOKENS_DEFAULT
    try:
        text = vlm.run_vlm(prompt=req.query, dynamic_prompt="", chart=img, max_new_tokens=max_new_tokens)
        print(f"Model answered: {text}")
        return {"text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"VLM inference failed: {e}")


@app.get("/health", status_code=200)
def health():
    """
    Health check endpoint.

    Returns:
        A JSON object indicating service health.
    """
    return {"health": "Ok"} 
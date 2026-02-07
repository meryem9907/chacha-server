from db import create_db_and_tables, AdminUser, create_admin_if_not_exists
from auth import create_access_token, authenticate_user, get_current_user, get_password_hash
from pwdlib import PasswordHash
from fastapi import FastAPI, UploadFile, File, Form,  Depends,HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import  OAuth2PasswordRequestForm
import os, logging, uuid, httpx, base64
from dotenv import load_dotenv
import uvicorn
from contextlib import asynccontextmanager
from config import Token, User
from datetime import timedelta
from typing import Annotated

load_dotenv(".env")
VLM_URL = os.getenv("VLM_URL", "http://vlm:5001/vlm/generate")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler for startup/shutdown behavior.

    This context manager runs at application startup to initialize the database
    schema and ensure an admin user exists before serving requests.

    Args:
        app: The FastAPI application instance.

    Yields:
        None
    """
    create_db_and_tables()
    pw = os.getenv("PW")
    admin_user = AdminUser(id=1, username="vqa-user", hashed_password=get_password_hash(pw))
    create_admin_if_not_exists(admin_user)
    yield
    

app = FastAPI(lifespan=lifespan)

origins = [f"{os.getenv("HOST_IP")}", "http://localhost"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(
level=logging.DEBUG,
format="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d - %(message)s"
)
logger = logging.getLogger("mylogger")

tmp_chart_dir = os.path.join(os.getcwd(),"charts/")
if (os.path.exists(tmp_chart_dir)==False):
    os.makedirs(os.path.join(os.getcwd(),"charts/"))


@app.post("/auth/token", response_model=Token)
def login(form: OAuth2PasswordRequestForm = Depends()):
    """
    Authenticate a user and return an access token.

    This endpoint validates user credentials using the configured authentication
    backend. If successful, it returns a bearer token with an expiration based
    on the configured environment settings.

    Args:
        form: OAuth2 password form containing username and password.

    Returns:
        A Token object containing an access token.

    Raises:
        HTTPException: If authentication fails with invalid credentials.
    """
    user = authenticate_user(form.username, form.password)
    if not user:
       raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_access_token(
        subject=user.username,
        expires_delta=timedelta(minutes=float(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES"))),
    )
    return Token(access_token=token) 

@app.post("/vlm/query")
async def query_vlm( 
    current_user: Annotated[User, Depends(get_current_user)],
    query: str = Form(...), # ellipsis "..." signals a required field
    chart_photo: UploadFile = File(...),
    max_new_tokens: int = Form(128),
):
    """
    Submit a chart image and query to the VLM service and return the response.

    This endpoint requires authentication. It accepts a multipart/form-data
    payload containing a text query and an uploaded chart image. The image is
    read into memory, base64-encoded, and forwarded to an upstream VLM service.
    The generated text response is returned.

    Args:
        current_user: The authenticated user derived from the request context.
        query: Natural-language prompt/question to ask the model about the chart.
        chart_photo: Uploaded image file containing the chart.
        max_new_tokens: Maximum number of tokens to generate at the VLM service.

    Returns:
        The generated text response from the VLM service.

    Raises:
        HTTPException: If the upload is empty, the upstream request fails, or the
            VLM service returns a non-200 response.
    """
    # 1) Read bytes from UploadFile
    img_bytes = await chart_photo.read()
    if not img_bytes:
        raise HTTPException(status_code=400, detail="Empty upload")

    # 2) Determine extension/mimetype 
    # chart_photo.content_type e.g. "image/png"
    extension = "png"
    if chart_photo.content_type and "/" in chart_photo.content_type:
        extension = chart_photo.content_type.split("/")[-1].lower()

    # 3) Base64 encode (no data: prefix, just raw base64 string)
    image_b64 = base64.b64encode(img_bytes).decode("utf-8")
    payload = {"query": query, "image_b64": image_b64, "extension": extension, "max_new_tokens": max_new_tokens,}
    response= ""
    try:
        async with httpx.AsyncClient(timeout=300) as client:
            response = await client.post(VLM_URL, json=payload)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"VLM service request failed: {e}")

    if response.status_code != 200:
        raise HTTPException(status_code=500, detail=f"VLM error {response.status_code}: {response.text}")

    return response.json()["text"]

@app.get("/health", status_code=200)
def health():
    """
    Health check endpoint.

    Returns:
        A JSON object indicating service health.
    """
    return {"health": "Ok"} 


if __name__ == "__main__":
    """
    Entrypoint for running the application with Uvicorn.

    This block starts the FastAPI application using Uvicorn when the module is
    executed as a script.
    """
    uvicorn.run("main:app", port=5000, log_level="info")

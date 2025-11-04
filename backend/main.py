from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from datetime import datetime
from services.user_upload import UserUploadService


app = FastAPI(
    title="AI Music Mentor API",
    description="RAG-based music production feedback API",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"message": "AI Music Mentor API", "status": "running"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/upload_tracks")
async def upload_tracks(
    input_file: UploadFile,
    reference_file: UploadFile,
    genre: str = Form(...),
    stage: str = Form(...),
    user_prompt: str = Form(...),
):
    """Upload and process both input and reference tracks"""

    # Create session directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_id = f"session_{timestamp}"
    uploads_dir = Path("data/uploads")
    session_dir = uploads_dir / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    # Initialize service
    upload_service = UserUploadService()

    # Read file contents
    input_content = await input_file.read()
    ref_content = await reference_file.read()

    # Process both files using the service
    input_data = upload_service.process_and_save_file(
        input_content, input_file.filename, "input", session_dir
    )
    ref_data = upload_service.process_and_save_file(
        ref_content, reference_file.filename, "reference", session_dir
    )

    # Return response
    return {
        "session_id": session_id,
        "genre": genre,
        "stage": stage,
        "user_prompt": user_prompt,
        "input_track": input_data,
        "reference_track": ref_data,
    }

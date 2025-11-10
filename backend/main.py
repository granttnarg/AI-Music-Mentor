from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from datetime import datetime
import logging
from services.user_upload import UserUploadService
from services.prompt_loader import PromptLoader
from services.audio_rag import create_llm_chain, AudioRAG
from db.connection import get_database

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TOP_K = 3


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
    """
    Root endpoint.

    Returns:
        API name and status
    """
    return {"message": "AI Music Mentor API", "status": "running"}


@app.get("/health")
def health_check():
    """
    Health check endpoint.

    Returns:
        Service health status
    """
    return {"status": "healthy"}


@app.post("/upload_tracks")
async def upload_tracks(
    input_file: UploadFile,
    reference_file: UploadFile,
    genre: str = Form(...),
    stage: str = Form(...),
    user_prompt: str = Form(...),
):
    """
    Upload and process both input and reference tracks, then save to database.

    Args:
        input_file: User's input audio track file
        reference_file: Reference audio track file for comparison
        genre: Music genre of the tracks
        stage: Production stage (mixing, mastering, etc.)
        user_prompt: User's question or description

    Returns:
        Session metadata, processed track data, and upload_id from database
    """
    try:
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

        # Check if processing was successful
        if not input_data.get("success") or not ref_data.get("success"):
            return {
                "session_id": session_id,
                "genre": genre,
                "stage": stage,
                "user_prompt": user_prompt,
                "input_track": input_data,
                "reference_track": ref_data,
                "upload_id": None,
                "success": False,
            }

        # Save to database
        logger.info(f"Saving upload to database for session {session_id}")
        db_ops = get_database()
        upload_id = db_ops.add_user_upload(
            input_track_path=input_data["file_path"],
            ref_track_path=ref_data["file_path"],
            input_duration=input_data["duration"],
            input_sample_rate=input_data["sample_rate"],
            input_embedding=input_data["embedding"],
            ref_duration=ref_data["duration"],
            ref_sample_rate=ref_data["sample_rate"],
            ref_embedding=ref_data["embedding"],
            user_prompt=user_prompt,
            stage=stage,
            genre=genre,
            session_id=session_id,
            input_file_size_bytes=input_data["file_size_bytes"],
            reference_file_size_bytes=ref_data["file_size_bytes"],
            input_original_filename=input_data["original_filename"],
            reference_original_filename=ref_data["original_filename"],
            input_global_features=input_data["global_features"],
            ref_global_features=ref_data["global_features"],
        )
        logger.info(f"Upload saved with ID: {upload_id}")

        # Return response with upload_id
        return {
            "session_id": session_id,
            "genre": genre,
            "stage": stage,
            "user_prompt": user_prompt,
            "input_track": input_data,
            "reference_track": ref_data,
            "upload_id": upload_id,
            "success": True,
        }

    except Exception as e:
        logger.error(f"Error in upload_tracks: {str(e)}", exc_info=True)
        return {
            "session_id": session_id if "session_id" in locals() else None,
            "success": False,
            "error": str(e),
        }


@app.post("/feedback")
async def feedback(
    user_upload_id: int = Form(...),
    question: str = Form(...),
    k: int = Form(TOP_K),
):
    """
    Generate AI feedback for a user upload using RAG.

    Args:
        user_upload_id: ID of the user upload to generate feedback for
        question: User's question about their track
        k: Number of similar examples to retrieve (default: 3)

    Returns:
        Generated feedback text and success status
    """
    try:
        logger.info(
            f"Starting feedback generation for upload_id={user_upload_id}, question='{question}', k={k}"
        )

        db_ops = get_database()
        prompts = PromptLoader._load_prompts()
        llm_chain = create_llm_chain(prompts)
        rag_service = AudioRAG(db_ops, prompts, llm_chain)

        logger.info("RAG service initialized, generating feedback...")
        feedback_text = rag_service.generate_feedback(
            user_upload_id=user_upload_id,
            question=question,
            k=k,
        )

        logger.info(
            f"Feedback generated successfully. Length: {len(feedback_text) if feedback_text else 0} chars"
        )
        logger.info(
            f"Feedback preview: {feedback_text[:200] if feedback_text else 'None'}..."
        )

        # Ensure feedback_text is a string for JSON serialization
        feedback_str = str(feedback_text) if feedback_text is not None else ""

        return {"feedback": feedback_str, "success": True}

    except Exception as e:
        logger.error(f"Error generating feedback: {str(e)}", exc_info=True)
        return {"feedback": None, "success": False, "error": str(e)}

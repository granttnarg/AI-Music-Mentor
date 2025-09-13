import streamlit as st
from pathlib import Path
from datetime import datetime
from src.audio_features import AudioFeatureService
from db.db import AudioRAGDatabase
from db.operations import AudioRAGOperations
from services.audio_rag import AudioRAG
from dotenv import load_dotenv
import os

GENRES = [
    "deep techno",
    "hard techno",
    "broken techno",
    "tech-House",
    "house",
    "electro",
    "vocal techno",
    "ambient",
    "other",
]

load_dotenv()


# Initialize database connection
@st.cache_resource
def get_database():
    """Initialize and return database connection"""
    connection_url = os.getenv(
        "DB_CONNECTION_URL", "postgresql://postgres:<ADD_TOENV_FILE>"
    )
    db = AudioRAGDatabase(connection_url)
    # db.reset_database()
    # db.setup_database()
    return AudioRAGOperations(db)


def process_and_save_file(
    file, file_type, session_dir, session_id, dropdown_option, text_input
):
    """Process and save a single file - now returns processed audio data"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    clean_name = Path(file.name).stem
    new_file_info = f"{file_type}--{clean_name}--{timestamp}"
    file_path = session_dir / f"{new_file_info}.mp3"

    # Save the MP3 file
    with open(file_path, "wb") as f:
        f.write(file.getbuffer())

    # Process audio features
    service = AudioFeatureService()
    try:
        global_features = service.load_audio_file(file_path).extract_global_features(
            max_duration=150
        )
        embedding = service.create_embedding_vector(global_features)
        feature_data = service.build_feature_data_object(
            global_features, ["rhythm", "energy"]
        )

        return {
            "file_path": str(file_path),
            "original_filename": file.name,
            "file_size_bytes": file.size,
            "duration": feature_data["metadata"]["duration"],
            "sample_rate": feature_data["metadata"]["sample_rate"],
            "embedding": embedding,
            "success": True,
        }
    except Exception as e:
        st.error(f"Error processing audio: {e} for: {file.name}")
        return {"success": False, "error": str(e)}


# START OF UPLOAD
uploads_dir = Path("uploads")
uploads_dir.mkdir(exist_ok=True)

st.title("AI Music Mentor Dashboard")
st.markdown(
    "#### Upload your unfinished track to get helpful advice on how to finish it."
)
st.caption(
    "Our AI system is built from data of one experienced producers' feedback and gives advice in their subjective tone and style."
)


track_genre = st.selectbox("Unfinished Track Genre:", GENRES)
input_file = st.file_uploader("Upload Unfinished track - MP3 file", type=["mp3"])

if input_file:
    st.audio(input_file)

ref_file = st.file_uploader(
    "Upload Reference track, something your aiming to get closer to - MP3 file",
    type=["mp3"],
)

if ref_file:
    st.audio(ref_file)

text_input = st.text_input("What do you need help with on your track?:")
dropdown_option = st.selectbox(
    "Stage your track is at:", ["Sketch", "Half Finished", "Almost Finished"]
)


if st.button("Submit"):
    if input_file and ref_file is not None and text_input:
        # Create a session-specific folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_id = f"session_{timestamp}"
        session_dir = uploads_dir / session_id
        session_dir.mkdir(exist_ok=True)

        st.success(f"Created session folder: {session_dir}")

        # Process both files
        input_data = process_and_save_file(
            input_file, "input", session_dir, session_id, dropdown_option, text_input
        )
        ref_data = process_and_save_file(
            ref_file, "reference", session_dir, session_id, dropdown_option, text_input
        )

        if input_data["success"] and ref_data["success"]:
            # Save to database
            try:
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
                    user_prompt=text_input,
                    stage=dropdown_option,
                    genre=track_genre,
                    session_id=session_id,
                    input_file_size_bytes=input_data["file_size_bytes"],
                    reference_file_size_bytes=ref_data["file_size_bytes"],
                    input_original_filename=input_data["original_filename"],
                    reference_original_filename=ref_data["original_filename"],
                )

                st.success(f"✅ Successfully saved to database! Upload ID: {upload_id}")
                st.success(f"📁 Files saved in: {session_dir}")

                # Show summary
                st.subheader("Upload Summary")
                summary = {
                    "upload_id": upload_id,
                    "session_id": session_id,
                    "user_question": text_input,
                    "stage": dropdown_option,
                    "input_file": input_data["original_filename"],
                    "reference_file": ref_data["original_filename"],
                }
                st.json(summary)

                # Generate AI feedback using RAG
                st.subheader("🎵 AI Music Mentor Feedback")
                with st.spinner("Analyzing your track and generating feedback..."):
                    try:
                        # Create RAG service using existing database connection
                        rag_service = AudioRAG(db_ops.db)

                        # Generate feedback using the upload ID
                        feedback = rag_service.generate_feedback(
                            user_upload_id=upload_id,
                            question=text_input,
                            k=3,  # Get top 3 similar examples
                        )

                        st.markdown(feedback)

                    except Exception as e:
                        st.error(f"❌ Could not generate feedback: {e}")
                        st.info(
                            "💡 This might be because there are no training examples in the database yet."
                        )

            except Exception as e:
                st.error(f"❌ Database error: {e}")

        else:
            st.error("Failed to process one or both audio files")

    else:
        if not input_file:
            st.warning("Please upload an input track")
        if not ref_file:
            st.warning("Please upload a reference track")
        if not text_input:
            st.warning("Please enter what you need help with")

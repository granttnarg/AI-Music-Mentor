import streamlit as st
from pathlib import Path
from datetime import datetime
from src.audio_features import AudioFeatureService
from db.db import AudioRAGDatabase
from db.operations import AudioRAGOperations
from dotenv import load_dotenv
import os

load_dotenv()


def process_and_save_training_file(file, file_type, session_dir):
    """Process and save a training file - returns processed audio data"""
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
            "duration": feature_data["metadata"]["duration"],
            "sample_rate": feature_data["metadata"]["sample_rate"],
            "embedding": embedding,
            "success": True,
        }
    except Exception as e:
        st.error(f"Error processing audio: {e} for: {file.name}")
        return {"success": False, "error": str(e)}


@st.cache_resource
def get_database():
    """Initialize and return database connection"""
    connection_url = os.getenv(
        "DB_CONNECTION_URL", "postgresql://postgres:<ADD_TOENV_FILE>"
    )
    db = AudioRAGDatabase(connection_url)
    return AudioRAGOperations(db)


def show_add_new_tab():
    """Show the Add New Training Example tab content"""
    st.markdown("#### Add feedback entries to the RAG database")
    st.caption(
        "Upload track pairs and provide expert feedback for training the AI system"
    )

    GENRES = [
        "techno",
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

    # Create two columns for the audio uploads
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Track")
        input_file = st.file_uploader(
            "Upload input track (unfinished)", type=["mp3"], key="input"
        )
        if input_file:
            st.audio(input_file)

    with col2:
        st.subheader("Reference Track")
        ref_file = st.file_uploader(
            "Upload reference track (finished/target)", type=["mp3"], key="reference"
        )
        if ref_file:
            st.audio(ref_file)

    # Track metadata
    st.subheader("Track Information")
    track_stage = st.selectbox(
        "Input track stage:", ["Sketch", "Half Finished", "Almost Finished"]
    )

    # Feedback sections
    st.subheader("Feedback Content")

    # General comments
    st.markdown("**General Comments**")
    track_genre = st.selectbox("Genre:", GENRES)

    general_comments = st.text_area(
        "Overall feedback and observations",
        placeholder="Provide general observations about the track, overall direction, strengths and areas for improvement...",
        height=100,
    )

    # Create two columns for feedback types
    feedback_col1, feedback_col2 = st.columns(2)

    with feedback_col1:
        st.markdown("**Rhythmic Feedback**")
        rhythmic_feedback = st.text_area(
            "Rhythm analysis and suggestions",
            placeholder="Comments on groove, timing, percussion patterns, rhythmic complexity...",
            height=120,
            key="rhythm_feedback",
        )

        rhythmic_practical = st.text_area(
            "Rhythmic practical suggestions",
            placeholder="Specific exercises or techniques to improve rhythmic elements...",
            height=100,
            key="rhythm_practical",
        )

    with feedback_col2:
        st.markdown("**EQ Feedback**")
        eq_feedback = st.text_area(
            "EQ and frequency balance analysis",
            placeholder="Comments on frequency balance, EQ issues, spectral analysis...",
            height=120,
            key="eq_feedback",
        )

        eq_practical = st.text_area(
            "EQ practical suggestions",
            placeholder="Specific EQ techniques, frequency targeting, mixing advice...",
            height=100,
            key="eq_practical",
        )

    # Preview section
    st.subheader("Entry Preview")
    if st.button("Generate Preview"):
        if input_file and ref_file:
            # Dynamically collect all feedback that has content
            feedback_data = {}

            # Check each feedback field and add to preview if filled
            if general_comments.strip():
                feedback_data["general_comments"] = general_comments.strip()

            if rhythmic_feedback.strip():
                feedback_data["rhythmic_feedback"] = rhythmic_feedback.strip()

            if rhythmic_practical.strip():
                feedback_data["rhythmic_practical"] = rhythmic_practical.strip()

            if eq_feedback.strip():
                feedback_data["eq_feedback"] = eq_feedback.strip()

            if eq_practical.strip():
                feedback_data["eq_practical"] = eq_practical.strip()

            preview_data = {
                "timestamp": datetime.now().isoformat(),
                "input_track": {
                    "filename": input_file.name,
                    "stage": track_stage,
                    "genre": track_genre,
                },
                "reference_track": {"filename": ref_file.name},
                "feedback": feedback_data,
                "feedback_count": len(feedback_data),
            }

            st.json(preview_data)
        else:
            st.warning("Please upload both input and reference tracks")

    # Action buttons
    st.subheader("Actions")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Save Entry", type="primary"):
            if input_file and ref_file and general_comments:
                # Create training session directory
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                session_id = f"training_{timestamp}"
                uploads_dir = Path("uploads/training_entries")
                uploads_dir.mkdir(exist_ok=True)
                session_dir = uploads_dir / session_id
                session_dir.mkdir(exist_ok=True)

                st.info(f"Processing files in session: {session_id}")

                # Process both files
                input_data = process_and_save_training_file(
                    input_file, "input", session_dir
                )
                ref_data = process_and_save_training_file(
                    ref_file, "reference", session_dir
                )

                if input_data["success"] and ref_data["success"]:
                    # Prepare feedback items for database
                    feedback_items = []

                    if general_comments.strip():
                        feedback_items.append(
                            {
                                "feedback_type": "general",
                                "feedback_text": general_comments.strip(),
                            }
                        )

                    if rhythmic_feedback.strip():
                        feedback_items.append(
                            {
                                "feedback_type": "rhythm",
                                "feedback_text": rhythmic_feedback.strip(),
                            }
                        )

                    if rhythmic_practical.strip():
                        feedback_items.append(
                            {
                                "feedback_type": "rhythm_practical",
                                "feedback_text": rhythmic_practical.strip(),
                            }
                        )

                    if eq_feedback.strip():
                        feedback_items.append(
                            {
                                "feedback_type": "eq",
                                "feedback_text": eq_feedback.strip(),
                            }
                        )

                    if eq_practical.strip():
                        feedback_items.append(
                            {
                                "feedback_type": "eq_practical",
                                "feedback_text": eq_practical.strip(),
                            }
                        )

                    # Save to database
                    try:
                        db_ops = get_database()
                        training_id = db_ops.add_training_example(
                            input_track_path=input_data["file_path"],
                            ref_track_path=ref_data["file_path"],
                            input_duration=input_data["duration"],
                            input_sample_rate=input_data["sample_rate"],
                            input_embedding=input_data["embedding"],
                            ref_duration=ref_data["duration"],
                            ref_sample_rate=ref_data["sample_rate"],
                            ref_embedding=ref_data["embedding"],
                            feedback_items=feedback_items,
                            genre=track_genre,
                        )

                        st.success(f"✅ Training example saved! ID: {training_id}")
                        st.success(f"📁 Files saved in: {session_dir}")
                        st.info(f"💬 Added {len(feedback_items)} feedback items")

                    except Exception as e:
                        st.error(f"❌ Database error: {e}")

                else:
                    st.error("Failed to process one or both audio files")

            else:
                st.error(
                    "Please fill in required fields: both audio files and general comments"
                )

    with col2:
        if st.button("Clear Form"):
            st.rerun()

    with col3:
        if st.button("Export All Entries"):
            st.info("Export functionality coming soon...")

    # Footer info
    st.markdown("---")
    st.caption(
        "This admin interface helps build the RAG database with expert feedback examples"
    )

import streamlit as st
from pathlib import Path
from datetime import datetime
from src.audio_features import AudioFeatureService
import json
from pprint import pprint
import numpy as np


# We need to serialize our data to save it as json.
def numpy_serializer(obj):
    """Convert numpy types to Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def process_and_save_file(
    file, file_type, session_dir, session_id, dropdown_option, text_input
):
    """Process and save a single file with its metadata"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    clean_name = Path(file.name).stem  # Remove extension
    new_file_info = f"{file_type}--{clean_name}--{timestamp}"
    file_path = session_dir / f"{new_file_info}.mp3"

    with open(file_path, "wb") as f:
        f.write(file.getbuffer())
    upload_time = datetime.now().isoformat()

    # Create metadata for JSON sidecar
    service = AudioFeatureService()
    global_features = "N/A"
    try:
        global_features = service.load_audio_file(file_path).extract_global_features(
            max_duration=150
        )  # Set to a short time so we can TEST. REMOVE THIS LATER
    except Exception as e:
        st.error(f"Error processing audio type : {e} for: {file.name}")
        return None, None

    metadata = {
        "original_filename": file.name,
        "file_type": file_type,  # "input" or "reference"
        "file_path": str(file_path),
        "upload_datetime": upload_time,
        "stage": dropdown_option,
        "user_question": text_input,
        "file_size_bytes": file.size,
        "session_id": session_id,
        "processed": {
            "global_feature_embedding": service.create_embedding_vector(
                global_features
            ),
            "global_feature_data": service.build_feature_data_object(
                global_features, ["rhythm", "energy"]
            ),
            "processed_at": datetime.now().isoformat(),
        },
    }

    # Save JSON sidecar file
    json_path = session_dir / f"{new_file_info}.mp3.json"
    with open(json_path, "w") as f:
        json.dump(metadata, f, default=numpy_serializer, indent=2)

    return file_path, json_path


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


input_file = st.file_uploader("Upload Unfinished track - MP3 file", type=["mp3"])
ref_file = st.file_uploader(
    "Upload Reference track, something your aiming to get closer to - MP3 file",
    type=["mp3"],
)
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

        # Process input file
        input_file_path, input_json_path = process_and_save_file(
            input_file, "input", session_dir, session_id, dropdown_option, text_input
        )

        if input_file_path:
            st.success(f"Input file uploaded: {input_file_path}")
            st.success(f"Input metadata saved: {input_json_path}")

        # Process reference file
        ref_file_path, ref_json_path = process_and_save_file(
            ref_file, "reference", session_dir, session_id, dropdown_option, text_input
        )

        if ref_file_path:
            st.success(f"Reference file uploaded: {ref_file_path}")
            st.success(f"Reference metadata saved: {ref_json_path}")

        # Create a session summary JSON
        session_metadata = {
            "session_id": session_id,
            "created_at": datetime.now().isoformat(),
            "user_question": text_input,
            "stage": dropdown_option,
            "files": {
                "input_track": {
                    "original_name": input_file.name,
                    "saved_path": str(input_file_path) if input_file_path else None,
                    "metadata_path": str(input_json_path) if input_json_path else None,
                },
                "reference_track": {
                    "original_name": ref_file.name,
                    "saved_path": str(ref_file_path) if ref_file_path else None,
                    "metadata_path": str(ref_json_path) if ref_json_path else None,
                },
            },
        }

        session_json_path = session_dir / "session_info.json"
        with open(session_json_path, "w") as f:
            json.dump(session_metadata, f, indent=2)

        st.success("Form submitted successfully!")
        st.code(f"Session folder: {session_dir}")

        # Show the session metadata
        st.subheader("Session Summary")
        st.json(session_metadata)

    else:
        if not input_file:
            st.warning("Please upload an input track")
        if not ref_file:
            st.warning("Please upload a reference track")
        if not text_input:
            st.warning("Please enter what you need help with")

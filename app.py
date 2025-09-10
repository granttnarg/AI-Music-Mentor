import streamlit as st
from pathlib import Path
from datetime import datetime
import json

uploads_dir = Path("uploads")
uploads_dir.mkdir(exist_ok=True)

st.title("AI Music Mentor Dashboard")
st.markdown(
    "#### Upload your unfinished track to get helpful advice on how to finish it."
)
st.caption(
    "Our AI system is built from data of one experienced producers' feedback and gives advice in their subjective tone and style."
)

# Basic UI for loads of 1 file
uploaded_file = st.file_uploader("Upload Unfinished track - MP3 file", type=["mp3"])
text_input = st.text_input("What do you need help with on your track?:")
dropdown_option = st.selectbox(
    "Stage your track is at:", ["Sketch", "Half Finished", "Almost Finished"]
)


if st.button("Submit"):
    file_path = None
    json_path = None

    if uploaded_file is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = uploads_dir / f'{timestamp}--{uploaded_file.name}'

        # Save the MP3 file
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Create metadata for JSON sidecar
        # TODO: We should extend this to a DB entry later

        metadata = {
            "original_filename": uploaded_file.name,
            "upload_timestamp": timestamp,
            "upload_datetime": datetime.now().isoformat(),
            "stage": dropdown_option,
            "user_question": text_input,
            "file_size_bytes": uploaded_file.size,
            "file_path": str(file_path)
        }

        # Save JSON sidecar file
        json_path = uploads_dir / f'{timestamp}--{uploaded_file.name}.json'
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        st.success(f"File uploaded and saved: {file_path}")
        st.success(f"Metadata saved: {json_path}")
    else:
        st.warning("No file uploaded")

    if text_input:
        st.write(f"Text entered: {text_input}")
    else:
        st.warning("No text entered")

    st.write(f"Selected option: {dropdown_option}")

    if uploaded_file is not None and text_input:
        st.success("Form submitted successfully!")
        st.code(f"Audio file: {file_path}")
        st.code(f"Metadata file: {json_path}")

        # Show the metadata content
        if json_path and json_path.exists():
            with open(json_path, 'r') as f:
                metadata = json.load(f)
            st.json(metadata)
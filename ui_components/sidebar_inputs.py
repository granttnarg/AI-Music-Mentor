import streamlit as st
import base64
from typing import Dict, Any


def render_sidebar(genres: list) -> Dict[str, Any]:
    """
    Render sidebar with track upload inputs.

    Args:
        genres: List of available music genres

    Returns:
        Dictionary containing all sidebar input values:
        - input_file: Uploaded input track file
        - ref_file: Uploaded reference track file
        - track_genre: Selected genre
        - text_input: User's question/description
        - dropdown_option: Track stage selection
        - visual_only: Visual-only mode toggle
        - submit_button: Whether submit was clicked
        - required_inputs: Boolean if all required fields filled
    """
    # Header image
    st.markdown(
        """
        <div style="height: 380px; overflow: hidden; border-radius: 8px; margin-bottom: 1rem;">
            <img src="data:image/jpeg;base64,{}" style="width: 100%; height: 100%; object-fit: cover;">
        </div>
        """.format(
            base64.b64encode(
                open("static/images/pexels-muffinlandge-27007091.jpg", "rb").read()
            ).decode()
        ),
        unsafe_allow_html=True,
    )

    st.subheader("Track Upload")
    st.caption("Upload your tracks and set preferences")

    # Genre selection
    track_genre = st.selectbox("Unfinished Track Genre:", genres)

    # Input track upload
    input_file = st.file_uploader(
        "Upload Unfinished track",
        type=["mp3", "wav", "aif"],
        help="MP3, WAV, or AIF file",
    )

    if input_file:
        st.audio(input_file, format="audio/mp3")

    # Reference track upload
    ref_file = st.file_uploader(
        "Upload Reference track",
        type=["mp3", "wav", "aif"],
        help="Something you're aiming to get closer to",
    )

    if ref_file:
        st.audio(ref_file, format="audio/mp3")

    # User question/description
    text_input = st.text_area(
        "What do you need help with?",
        height=80,
        placeholder="Describe what you need help with on your track...",
    )

    # Track stage selection
    dropdown_option = st.selectbox(
        "Track Stage:", ["Sketch", "Half Finished", "Almost Finished"]
    )

    # Visual-only toggle
    visual_only = st.checkbox(
        "Visual Only",
        value=False,
        help="Skip AI feedback for faster segmentation testing",
    )

    # Submit button
    submit_button = st.button(
        "Analyze Tracks", type="primary", use_container_width=True
    )

    # Check if required inputs are filled
    required_inputs = bool(input_file and ref_file and text_input)

    return {
        "input_file": input_file,
        "ref_file": ref_file,
        "track_genre": track_genre,
        "text_input": text_input,
        "dropdown_option": dropdown_option,
        "visual_only": visual_only,
        "submit_button": submit_button,
        "required_inputs": required_inputs,
    }

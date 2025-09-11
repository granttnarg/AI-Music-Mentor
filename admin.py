import streamlit as st
from pathlib import Path
from datetime import datetime

st.set_page_config(page_title="RAG Database Admin", layout="wide")

st.title("RAG Database Admin")
st.markdown("#### Add feedback entries to the RAG database")
st.caption("Upload track pairs and provide expert feedback for training the AI system")

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

    rhythmic_practice = st.text_area(
        "Rhythmic practice suggestions",
        placeholder="Specific exercises or techniques to improve rhythmic elements...",
        height=100,
        key="rhythm_practice",
    )

with feedback_col2:
    st.markdown("**EQ Feedback**")
    eq_feedback = st.text_area(
        "EQ and frequency balance analysis",
        placeholder="Comments on frequency balance, EQ issues, spectral analysis...",
        height=120,
        key="eq_feedback",
    )

    eq_practice = st.text_area(
        "EQ practice suggestions",
        placeholder="Specific EQ techniques, frequency targeting, mixing advice...",
        height=100,
        key="eq_practice",
    )

# Preview section
st.subheader("Entry Preview")
if st.button("Generate Preview"):
    if input_file and ref_file:
        preview_data = {
            "timestamp": datetime.now().isoformat(),
            "input_track": {
                "filename": input_file.name,
                "stage": track_stage,
                "genre": track_genre,
            },
            "reference_track": {"filename": ref_file.name},
            "feedback": {
                "general_comments": general_comments,
                "rhythmic": {
                    "feedback": rhythmic_feedback,
                    "practice_suggestions": rhythmic_practice,
                },
                "eq": {"feedback": eq_feedback, "practice_suggestions": eq_practice},
            },
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
            st.success("Entry saved to RAG database!")
            # TODO: Implement actual saving logic
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

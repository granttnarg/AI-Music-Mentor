import streamlit as st
from pathlib import Path
import httpx
from db.connection import get_database
from dotenv import load_dotenv
import os
import random
from db.models import UserUpload
from utils.style_loader import load_css
from utils.arrangement_visualizer import prepare_arrangement_visualizations
from ui_components.sidebar_inputs import render_sidebar
from ui_components.feedback_display import render_feedback
from ui_components.visualization_display import render_arrangement_visualization

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


# API Helper Functions
def upload_tracks_to_api(input_file, ref_file, genre, stage, user_prompt):
    """
    Upload tracks to the FastAPI backend for processing.

    Returns:
        dict: API response with session_id and track data
    """
    with httpx.Client(timeout=300.0) as client:
        files = {
            "input_file": (input_file.name, input_file.getvalue(), "audio/mpeg"),
            "reference_file": (ref_file.name, ref_file.getvalue(), "audio/mpeg"),
        }
        data = {
            "genre": genre,
            "stage": stage,
            "user_prompt": user_prompt,
        }
        response = client.post(f"{API_BASE_URL}/upload_tracks", files=files, data=data)
        response.raise_for_status()
        return response.json()


def generate_feedback_from_api(upload_id, question, k=3):
    """
    Generate AI feedback for a user upload via the FastAPI backend.

    Returns:
        dict: API response with feedback text and success status
    """
    with httpx.Client(timeout=300.0) as client:
        data = {
            "user_upload_id": upload_id,
            "question": question,
            "k": k,
        }
        response = client.post(f"{API_BASE_URL}/feedback", data=data)
        response.raise_for_status()
        return response.json()


# Constants
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

MUSIC_TIPS = [
    "**Tip**: Line up a reference track next to your track in your DAW to help you understand arrangement flow while your make your song. ",
    "**Did you know?**: The longer you spent on a track without resting breaks the more likely you are to lose touch with how it sounds. Don't forget to give your ears and mind a rest when creating!",
    "**Pro tip**: Make sure your kick and bass fundemntal frequency is at least 50Hz apart to avoid large overlap that ruin the power and clarity in your low end.",
    "**Technique**: Side-chain compression can create that classic pumping effect, and when done subtly is also great to give space for other elements in the mix.",
    "**Energy tip**: Build tension with low volume subtle rising elements before your main drops can help lead the listener making impact more rewarding.",
    "**Arrangement**: Leave space in your mix - not every element needs to play at once, its often best to let one or two elements show off in each section.",
    "**Variation**: If you change multiple elements in A section at once it can breathe a whole new life into your tracks progression and is more impactful that just adding in hi hats or a drum variation.",
    "**Focus**: A great track usually has one main element that everything else supports, try to understand which element is the strongest and use that as your songs backbone.",
]

# Initialize
load_dotenv()
load_css()

# START OF UPLOAD
uploads_dir = Path("data/uploads")
uploads_dir.mkdir(exist_ok=True)

st.title("AI Music Mentor")
st.markdown("**techno edition**")
st.subheader("Get unstuck. Get inspired. Get heard.")
st.markdown("Practical AI feedback in the style of Berlin Producer 16 Faces \n \n")

# Render sidebar and get user inputs
with st.sidebar:
    sidebar_inputs = render_sidebar(GENRES)

# Extract inputs for easier access
input_file = sidebar_inputs["input_file"]
ref_file = sidebar_inputs["ref_file"]
track_genre = sidebar_inputs["track_genre"]
text_input = sidebar_inputs["text_input"]
dropdown_option = sidebar_inputs["dropdown_option"]
visual_only = sidebar_inputs["visual_only"]
submit_button = sidebar_inputs["submit_button"]
required_inputs = sidebar_inputs["required_inputs"]

# Main content area
if not submit_button:
    # Welcome message in main area when no submission
    st.markdown(
        """
        #### Welcome to AI Music Mentor!

        Upload your unfinished track and a reference track using the sidebar, then get personalized feedback to help you finish your music.

        **How it works:**
        1. **Upload your tracks** - Your unfinished electronic music track and a reference track with a style and structure close to what your aiming for.
        2. **Set your track info** - Genre, stage, and a simple prompt of what specific help your after.
        3. **Get AI feedback** - Receive technical arrangement analysis and personalized practical production advice

        Our AI system is trained on experienced producer 16 Faces feedback style and gives production and arrangement feedback to get your to get you unstuck.
        """
    )
else:
    # For visual-only mode, don't require text input
    required_inputs = input_file and ref_file is not None
    if not visual_only:
        required_inputs = required_inputs and text_input

    if required_inputs:
        with st.spinner("Processing your tracks..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            tip_container = st.empty()

            current_tip = random.choice(MUSIC_TIPS)
            tip_container.info(current_tip)

            status_text.text("Uploading and processing tracks...")
            progress_bar.progress(20)

            try:
                # Call API to process both tracks and save to database
                api_result = upload_tracks_to_api(
                    input_file=input_file,
                    ref_file=ref_file,
                    genre=track_genre,
                    stage=dropdown_option,
                    user_prompt=text_input,
                )

                # Extract data from API response
                session_id = api_result.get("session_id")
                input_data = api_result.get("input_track", {})
                ref_data = api_result.get("reference_track", {})
                upload_id = api_result.get("upload_id")

                progress_bar.progress(60)
                current_tip = random.choice(MUSIC_TIPS)
                tip_container.info(current_tip)

                if api_result.get("success") and upload_id:
                    # Initialize variables that might be referenced in error handling
                    arrangement_error = None
                    input_track_data = None
                    ref_track_data = None
                    input_viz_fig = None
                    ref_viz_fig = None

                    # Collect all data before displaying anything
                    status_text.text("Generating arrangement analysis...")
                    progress_bar.progress(85)

                    try:
                        # Get track data with arrangement information from database
                        db_ops = get_database()
                        session = db_ops.db.get_session()
                        try:
                            upload = (
                                session.query(UserUpload)
                                .filter(UserUpload.id == upload_id)
                                .first()
                            )
                            if upload:
                                input_track_data = db_ops.get_track(
                                    int(upload.input_track_id)
                                )
                                ref_track_data = db_ops.get_track(
                                    int(upload.reference_track_id)
                                )

                                # Prepare arrangement visualizations
                                viz_result = prepare_arrangement_visualizations(
                                    input_track_data=input_track_data,
                                    ref_track_data=ref_track_data,
                                    input_filename=input_data["original_filename"],
                                    ref_filename=ref_data["original_filename"],
                                )

                                # Extract visualization figures
                                if viz_result.get("comparison_viz_fig"):
                                    comparison_viz_fig = viz_result[
                                        "comparison_viz_fig"
                                    ]
                                if viz_result.get("input_viz_fig"):
                                    input_viz_fig = viz_result["input_viz_fig"]
                                if viz_result.get("ref_viz_fig"):
                                    ref_viz_fig = viz_result["ref_viz_fig"]
                                if viz_result.get("error"):
                                    arrangement_error = viz_result["error"]

                        finally:
                            session.close()

                    except Exception as e:
                        arrangement_error = str(e)

                    # Generate AI feedback (skip if visual-only mode)
                    if "feedback" not in locals():
                        feedback = None
                    if "feedback_error" not in locals():
                        feedback_error = None

                    if not visual_only:
                        status_text.text("Generating AI feedback...")
                        progress_bar.progress(95)

                        try:
                            api_result = generate_feedback_from_api(
                                upload_id=upload_id,
                                question=text_input,
                                k=3,
                            )
                            feedback = api_result["feedback"]
                        except Exception as e:
                            feedback_error = str(e)
                    else:
                        # Skip AI feedback in visual-only mode
                        status_text.text("Visual-only mode - skipping AI feedback")
                        progress_bar.progress(95)

                    # Clear progress indicators
                    progress_bar.empty()
                    status_text.empty()
                    tip_container.empty()

                    # NOW DISPLAY EVERYTHING AT ONCE
                    st.success("Analysis Complete! Your tracks have been processed.")

                    # Create two-column layout: viz on left, feedback on right
                    viz_col, feedback_col = st.columns([3, 2])  # 60% viz, 40% feedback

                    with viz_col:
                        render_arrangement_visualization(
                            arrangement_error=arrangement_error,
                            comparison_viz_fig=locals().get("comparison_viz_fig"),
                            input_viz_fig=locals().get("input_viz_fig"),
                            ref_viz_fig=locals().get("ref_viz_fig"),
                            input_track_data=input_track_data,
                            ref_track_data=ref_track_data,
                            input_file=input_file,
                            ref_file=ref_file,
                        )

                    # AI Feedback column
                    with feedback_col:
                        render_feedback(feedback, feedback_error, visual_only)

                else:
                    # Clear progress indicators on error
                    progress_bar.empty()
                    status_text.empty()
                    tip_container.empty()
                    st.error("Failed to process one or both audio files")

            except Exception as e:
                # Clear progress indicators on error
                progress_bar.empty()
                status_text.empty()
                tip_container.empty()
                st.error(f"Error processing tracks: {e}")

    else:
        # Show missing input warnings in main area
        if not input_file:
            st.warning("Please upload an input track using the sidebar")
        if not ref_file:
            st.warning("Please upload a reference track using the sidebar")
        if not visual_only and not text_input:
            st.warning(
                "Please describe what you need help with (or enable Visual Only mode)"
            )

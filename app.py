import streamlit as st
from pathlib import Path
from datetime import datetime
import httpx
from db.db import AudioRAGDatabase
from db.operations import AudioRAGOperations
from services.audio_rag import AudioRAG
from services.song_visualizer_service import SongVisualizerService
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt
import base64
import random
import json
import numpy as np
from db.models import UserUpload
from src.classifier.arrangement_postprocessing import process_arrangement_predictions
from services.prompt_loader import PromptLoader
from services.audio_rag import create_llm_chain

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

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


# START OF UPLOAD
uploads_dir = Path("data/uploads")
uploads_dir.mkdir(exist_ok=True)

# Custom CSS for styling and layout
st.markdown(
    """
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Rubik+Iso&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global font styling */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main container styling */
    .block-container {
        max-width: 90% !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        padding-top: 2rem !important;
    }
    
    /* Main title styling */
    h1 {
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
        font-size: 3rem !important;
        color: #1e293b !important;
    }
    
    /* Subtitle styling */
    .stApp > div > div > div > div > div:nth-child(2) h3 {
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
        color: #475569 !important;
        margin-top: 0 !important;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        width: 25rem !important;
        background-color: #f8fafc !important;
        border-right: 1px solid #e2e8f0 !important;
    }
    
    /* Sidebar content styling */
    .stSidebar > div {
        background-color: #f8fafc !important;
        padding-top: 1rem !important;
    }
    
    /* Sidebar headers */
    .stSidebar h3 {
        color: #334155 !important;
        font-weight: 600 !important;
    }
    
    /* Sidebar captions */
    .stSidebar .caption {
        color: #e3dfd2 !important;
    }
    
    /* Primary button styling (Analyze Tracks button) */
    .stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%) !important;
        border: none !important;
        border-radius: 8px !important;
        color: white !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        padding: 0.75rem 1.5rem !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 2px 4px rgba(59, 130, 246, 0.2) !important;
    }
    
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(90deg, #2563eb 0%, #1d4ed8 100%) !important;
        box-shadow: 0 4px 8px rgba(59, 130, 246, 0.3) !important;
        transform: translateY(-1px) !important;
    }
    
    /* Adjust main content area */
    .css-18e3th9 {
        padding-left: 1rem !important;
    }
    
    /* Add horizontal padding to column containers */
    .stColumn > div {
        padding-left: 20px !important;
        padding-right: 20px !important;
    }
    
    /* Add padding to element containers within columns */
    .stColumn .stElementContainer {
        padding-left: 10px !important;
        padding-right: 10px !important;
    }
    
    /* Ensure visualization containers have proper spacing */
    .stColumn .st-emotion-cache-1vo6xi6 {
        padding-left: 10px !important;
        padding-right: 10px !important;
    }
    
    /* Main title styling with Rubik Iso font - only for main headings, not feedback content */
    #get-unstuck-get-inspired-get-heard {
        font-family: "Rubik Iso", system-ui !important;
        font-weight: 400 !important;
        font-size: 2rem !important;
        color: #1e293b !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Keep feedback headings with regular font */
    .stColumn h3, .stColumn h2, .stColumn h4 {
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        color: #1e293b !important;
    }
    
    /* Success message styling */
    .stAlert[data-baseweb="notification"][kind="success"] {
        background-color: #f0fdf4 !important;
        border: 1px solid #bbf7d0 !important;
        border-radius: 8px !important;
    }
    
    /* Info styling */
    .stAlert[data-baseweb="notification"][kind="info"] {
        background-color: #eff6ff !important;
        border: 1px solid #bfdbfe !important;
        border-radius: 8px !important;
    }
    
    /* File uploader styling */
    .stFileUploader > div {
        border-radius: 8px !important;
        border: 2px dashed #e2e8f0 !important;
    }
    
    .stFileUploader > div:hover {
        border-color: #3b82f6 !important;
    }
    

</style>
""",
    unsafe_allow_html=True,
)

st.title("AI Music Mentor")
st.markdown("**techno edition**")
st.subheader("Get unstuck. Get inspired. Get heard.")
st.markdown("Practical AI feedback in the style of Berlin Producer 16 Faces \n \n")

# Create layout with sidebar for inputs and main area for output
with st.sidebar:
    # Add John's image above Track Upload with custom container
    st.markdown(
        """
        <div style="height: 380px; overflow: hidden; border-radius: 8px; margin-bottom: 1rem;">
            <img src="data:image/jpeg;base64,{}" style="width: 100%; height: 100%; object-fit: cover;">
        </div>
        """.format(
            base64.b64encode(
                open("images/pexels-muffinlandge-27007091.jpg", "rb").read()
            ).decode()
        ),
        unsafe_allow_html=True,
    )

    st.subheader("Track Upload")
    st.caption("Upload your tracks and set preferences")

    track_genre = st.selectbox("Unfinished Track Genre:", GENRES)

    input_file = st.file_uploader(
        "Upload Unfinished track",
        type=["mp3", "wav", "aif"],
        help="MP3, WAV, or AIF file",
    )

    if input_file:
        st.audio(input_file, format="audio/mp3")

    ref_file = st.file_uploader(
        "Upload Reference track",
        type=["mp3", "wav", "aif"],
        help="Something you're aiming to get closer to",
    )

    if ref_file:
        st.audio(ref_file, format="audio/mp3")

    text_input = st.text_area(
        "What do you need help with?",
        height=80,
        placeholder="Describe what you need help with on your track...",
    )

    dropdown_option = st.selectbox(
        "Track Stage:", ["Sketch", "Half Finished", "Almost Finished"]
    )

    # Visual-only toggle for quick segmentation testing
    visual_only = st.checkbox(
        "Visual Only",
        value=False,
        help="Skip AI feedback for faster segmentation testing",
    )

    # Submit button in sidebar
    submit_button = st.button(
        "Analyze Tracks", type="primary", use_container_width=True
    )

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

            # Music production tips to show during loading
            music_tips = [
                "**Tip**: Line up a reference track next to your track in your DAW to help you understand arrangement flow while your make your song. ",
                "**Did you know?**: The longer you spent on a track without resting breaks the more likely you are to lose touch with how it sounds. Don't forget to give your ears and mind a rest when creating!",
                "**Pro tip**: Make sure your kick and bass fundemntal frequency is at least 50Hz apart to avoid large overlap that ruin the power and clarity in your low end.",
                "**Technique**: Side-chain compression can create that classic pumping effect, and when done subtly is also great to give space for other elements in the mix.",
                "**Energy tip**: Build tension with low volume subtle rising elements before your main drops can help lead the listener making impact more rewarding.",
                "**Arrangement**: Leave space in your mix - not every element needs to play at once, its often best to let one or two elements show off in each section.",
                "**Variation**: If you change multiple elements in A section at once it can breathe a whole new life into your tracks progression and is more impactful that just adding in hi hats or a drum variation.",
                "**Focus**: A great track usually has one main element that everything else supports, try to understand which element is the strongest and use that as your songs backbone.",
            ]

            current_tip = random.choice(music_tips)
            tip_container.info(current_tip)

            status_text.text("Uploading and processing tracks...")
            progress_bar.progress(20)

            try:
                # Call API to process both tracks
                with httpx.Client(timeout=300.0) as client:
                    files = {
                        "input_file": (
                            input_file.name,
                            input_file.getvalue(),
                            "audio/mpeg",
                        ),
                        "reference_file": (
                            ref_file.name,
                            ref_file.getvalue(),
                            "audio/mpeg",
                        ),
                    }
                    data = {
                        "genre": track_genre,
                        "stage": dropdown_option,
                        "user_prompt": text_input,
                    }

                    response = client.post(
                        f"{API_BASE_URL}/upload_tracks",
                        files=files,
                        data=data,
                    )
                    response.raise_for_status()
                    api_result = response.json()

                # Extract data from API response
                session_id = api_result["session_id"]
                input_data = api_result["input_track"]
                ref_data = api_result["reference_track"]

                progress_bar.progress(50)
                current_tip = random.choice(music_tips)
                tip_container.info(current_tip)

                if input_data["success"] and ref_data["success"]:
                    status_text.text("💾 Saving to database...")
                    progress_bar.progress(60)

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
                            input_global_features=input_data["global_features"],
                            ref_global_features=ref_data["global_features"],
                        )

                        # Collect all data before displaying anything
                        status_text.text("Generating arrangement analysis...")
                        progress_bar.progress(85)

                        # Prepare all visualization data
                        input_track_data = None
                        ref_track_data = None
                        input_viz_fig = None
                        ref_viz_fig = None
                        arrangement_error = None

                        try:
                            # Get track data with arrangement information
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

                                    # Pre-generate time-aligned comparison visualization
                                    visualizer = SongVisualizerService()

                                    input_blocks = None
                                    ref_blocks = None

                                    # Process input track
                                    if input_track_data and input_track_data.get(
                                        "raw_arrangement_pattern"
                                    ):
                                        if input_track_data.get(
                                            "raw_predictions"
                                        ) and input_track_data.get(
                                            "raw_confidence_scores"
                                        ):
                                            raw_predictions = json.loads(
                                                input_track_data["raw_predictions"]
                                            )
                                            confidence_scores = json.loads(
                                                input_track_data[
                                                    "raw_confidence_scores"
                                                ]
                                            )

                                            input_blocks, analysis = (
                                                process_arrangement_predictions(
                                                    np.array(raw_predictions),
                                                    np.array(confidence_scores),
                                                    ["O", "A", "B", "C"],
                                                    min_segment_length=2,
                                                    confidence_threshold=0.4,
                                                )
                                            )

                                    # Process reference track
                                    if ref_track_data and ref_track_data.get(
                                        "raw_arrangement_pattern"
                                    ):
                                        if ref_track_data.get(
                                            "raw_predictions"
                                        ) and ref_track_data.get(
                                            "raw_confidence_scores"
                                        ):
                                            raw_predictions = json.loads(
                                                ref_track_data["raw_predictions"]
                                            )
                                            confidence_scores = json.loads(
                                                ref_track_data["raw_confidence_scores"]
                                            )

                                            ref_blocks, analysis = (
                                                process_arrangement_predictions(
                                                    np.array(raw_predictions),
                                                    np.array(confidence_scores),
                                                    ["O", "A", "B", "C"],
                                                    min_segment_length=2,
                                                    confidence_threshold=0.4,
                                                )
                                            )

                                    # Create time-aligned comparison if both tracks have arrangement data
                                    if input_blocks and ref_blocks:
                                        comparison_viz_fig = visualizer.plot_time_aligned_comparison(
                                            input_audio_path=input_track_data[
                                                "file_path"
                                            ],
                                            input_arrangement_blocks=input_blocks,
                                            input_title=f"Input Track: {input_data['original_filename']}",
                                            reference_audio_path=ref_track_data[
                                                "file_path"
                                            ],
                                            reference_arrangement_blocks=ref_blocks,
                                            reference_title=f"Reference Track: {ref_data['original_filename']}",
                                        )
                                    else:
                                        # Fallback to individual plots if one track is missing arrangement data
                                        if input_blocks:
                                            input_viz_fig = visualizer.plot_arrangement_waveform(
                                                audio_path=input_track_data[
                                                    "file_path"
                                                ],
                                                arrangement_blocks=input_blocks,
                                                title=f"Input Track: {input_data['original_filename']}",
                                            )
                                        if ref_blocks:
                                            ref_viz_fig = visualizer.plot_arrangement_waveform(
                                                audio_path=ref_track_data["file_path"],
                                                arrangement_blocks=ref_blocks,
                                                title=f"Reference Track: {ref_data['original_filename']}",
                                            )

                            finally:
                                session.close()

                        except Exception as e:
                            arrangement_error = str(e)

                    except Exception as e:
                        progress_bar.empty()
                        status_text.empty()
                        tip_container.empty()
                        st.error(f"Database error: {e}")
                        feedback = None
                        feedback_error = None

                    # Generate AI feedback (skip if visual-only mode)
                    if "feedback" not in locals():
                        feedback = None
                    if "feedback_error" not in locals():
                        feedback_error = None

                    if not visual_only:
                        status_text.text("Generating AI feedback...")
                        progress_bar.progress(95)

                        try:
                            # Create RAG service using existing database connection
                            operations = AudioRAGOperations(db_ops.db)
                            prompts = PromptLoader._load_prompts()
                            llm_chain = create_llm_chain(prompts)
                            rag_service = AudioRAG(operations, prompts, llm_chain)
                            feedback = rag_service.generate_feedback(
                                user_upload_id=upload_id,
                                question=text_input,
                                k=3,  # Get top 3 similar examples
                            )
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

                    # Technical details commented out for cleaner demo UI
                    # with st.expander("Technical Details (click to expand)", expanded=False):
                    #     summary = {
                    #         "upload_id": upload_id,
                    #         "session_id": session_id,
                    #         "user_question": text_input,
                    #         "stage": dropdown_option,
                    #         "input_file": input_data["original_filename"],
                    #         "reference_file": ref_data["original_filename"],
                    #     }
                    #     st.json(summary)

                    # Create two-column layout: viz on left, feedback on right
                    viz_col, feedback_col = st.columns([3, 2])  # 60% viz, 40% feedback

                    with viz_col:
                        st.markdown("**Arrangement Analysis**")

                        if arrangement_error:
                            st.error(
                                f"Could not load arrangement visualizations: {arrangement_error}"
                            )
                        else:
                            # Time-aligned comparison visualization
                            if "comparison_viz_fig" in locals():
                                # Show patterns with BPM in compact format
                                pattern_col1, pattern_col2 = st.columns(2)
                                with pattern_col1:
                                    if input_track_data and input_track_data.get(
                                        "smoothed_arrangement_pattern"
                                    ):
                                        input_tempo = ""
                                        if input_track_data.get(
                                            "global_feature_data"
                                        ) and isinstance(
                                            input_track_data["global_feature_data"],
                                            dict,
                                        ):
                                            tempo = (
                                                input_track_data["global_feature_data"]
                                                .get("rhythm", {})
                                                .get("tempo")
                                            )
                                            if tempo:
                                                input_tempo = f" ({tempo:.0f} BPM)"
                                        st.markdown(
                                            f"**Input:** `{input_track_data['smoothed_arrangement_pattern']}`{input_tempo}"
                                        )

                                with pattern_col2:
                                    if ref_track_data and ref_track_data.get(
                                        "smoothed_arrangement_pattern"
                                    ):
                                        ref_tempo = ""
                                        if ref_track_data.get(
                                            "global_feature_data"
                                        ) and isinstance(
                                            ref_track_data["global_feature_data"], dict
                                        ):
                                            tempo = (
                                                ref_track_data["global_feature_data"]
                                                .get("rhythm", {})
                                                .get("tempo")
                                            )
                                            if tempo:
                                                ref_tempo = f" ({tempo:.0f} BPM)"
                                        st.markdown(
                                            f"**Reference:** `{ref_track_data['smoothed_arrangement_pattern']}`{ref_tempo}"
                                        )

                                # Full-width visualization
                                st.pyplot(comparison_viz_fig, use_container_width=True)
                                plt.close(comparison_viz_fig)

                                # Audio players underneath the visualization
                                audio_col1, audio_col2 = st.columns(2)
                                with audio_col1:
                                    st.markdown("**Input Track Audio:**")
                                    if input_file:
                                        st.audio(input_file, format="audio/mp3")
                                with audio_col2:
                                    st.markdown("**Reference Track Audio:**")
                                    if ref_file:
                                        st.audio(ref_file, format="audio/mp3")
                            else:
                                # Fallback to individual displays - show patterns first
                                pattern_col1, pattern_col2 = st.columns(2)
                                with pattern_col1:
                                    if input_track_data and input_track_data.get(
                                        "raw_arrangement_pattern"
                                    ):
                                        input_tempo = ""
                                        if input_track_data.get(
                                            "global_feature_data"
                                        ) and isinstance(
                                            input_track_data["global_feature_data"],
                                            dict,
                                        ):
                                            tempo = (
                                                input_track_data["global_feature_data"]
                                                .get("rhythm", {})
                                                .get("tempo")
                                            )
                                            if tempo:
                                                input_tempo = f" ({tempo:.0f} BPM)"
                                        st.markdown(
                                            f"**Input:** `{input_track_data['smoothed_arrangement_pattern']}`{input_tempo}"
                                        )
                                    else:
                                        st.info("Input pattern not available")

                                with pattern_col2:
                                    if ref_track_data and ref_track_data.get(
                                        "raw_arrangement_pattern"
                                    ):
                                        ref_tempo = ""
                                        if ref_track_data.get(
                                            "global_feature_data"
                                        ) and isinstance(
                                            ref_track_data["global_feature_data"], dict
                                        ):
                                            tempo = (
                                                ref_track_data["global_feature_data"]
                                                .get("rhythm", {})
                                                .get("tempo")
                                            )
                                            if tempo:
                                                ref_tempo = f" ({tempo:.0f} BPM)"
                                        st.markdown(
                                            f"**Reference:** `{ref_track_data['smoothed_arrangement_pattern']}`{ref_tempo}"
                                        )
                                    else:
                                        st.info("Reference pattern not available")

                                # Individual visualizations
                                viz_col1, viz_col2 = st.columns(2)
                                with viz_col1:
                                    if "input_viz_fig" in locals():
                                        st.pyplot(
                                            input_viz_fig, use_container_width=True
                                        )
                                        plt.close(input_viz_fig)
                                    st.markdown("**Input Track Audio:**")
                                    if input_file:
                                        st.audio(input_file, format="audio/mp3")

                                with viz_col2:
                                    if "ref_viz_fig" in locals():
                                        st.pyplot(ref_viz_fig, use_container_width=True)
                                        plt.close(ref_viz_fig)
                                    st.markdown("**Reference Track Audio:**")
                                    if ref_file:
                                        st.audio(ref_file, format="audio/mp3")

                    # AI Feedback column
                    with feedback_col:
                        st.markdown("**AI Music Mentor Feedback**")
                        if not visual_only:
                            if feedback_error:
                                st.error(
                                    f"Could not generate feedback: {feedback_error}"
                                )
                                st.info(
                                    "This might be because there are no training examples in the database yet."
                                )
                            else:
                                st.markdown(feedback)

                                # Add inspiration track section after feedback
                                # TODO: Re-enable for post-demo development
                                # try:
                                #     operations = AudioRAGOperations(db_ops.db)
                                #     prompts = PromptLoader._load_prompts()
                                #     llm_chain = create_llm_chain(prompts)
                                #     rag_service = AudioRAG(operations, prompts, llm_chain)
                                #     similar_examples, user_upload, retrieval_info = rag_service.retrieve_similar_examples(
                                #         user_upload_id=upload_id, k=1
                                #     )
                                #
                                #     if similar_examples and len(similar_examples) > 0:
                                #         top_example = similar_examples[0]
                                #         reference_track = top_example.get("reference_track")
                                #
                                #         # Only show if we have a reference track (fully formed track)
                                #         if reference_track:
                                #             st.markdown("---")
                                #             st.markdown("**🎵 Inspiration Track**")
                                #
                                #             track_name = Path(reference_track['file_path']).stem
                                #             arrangement_pattern = reference_track.get('arrangement_pattern', 'Unknown')
                                #
                                #             st.write(f"**Track:** {track_name}")
                                #             st.write(f"**Arrangement:** `{arrangement_pattern}`")
                                #             st.write("Check this fully-formed track for arrangement inspiration - it has a similar vibe to yours!")
                                #
                                #             # Add audio player for the inspiration track
                                #             try:
                                #                 if os.path.exists(reference_track['file_path']):
                                #                     with open(reference_track['file_path'], 'rb') as audio_file:
                                #                         audio_bytes = audio_file.read()
                                #                     st.audio(audio_bytes, format='audio/mp3')
                                #                 else:
                                #                     st.info("Audio file not found for this inspiration track")
                                #             except Exception as audio_error:
                                #                 st.info("Could not load audio for this track")
                                #
                                # except Exception as inspiration_error:
                                #     # Silently skip inspiration track if there's an error
                                #     pass
                        else:
                            st.success("Visual-only mode enabled")
                            st.info(
                                "AI feedback skipped for faster processing. Uncheck 'Visual Only' in the sidebar to get feedback."
                            )

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

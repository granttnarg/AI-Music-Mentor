import streamlit as st
from pathlib import Path
from datetime import datetime
from src.audio_features import AudioFeatureService
from db.db import AudioRAGDatabase
from db.operations import AudioRAGOperations
from services.audio_rag import AudioRAG
from services.song_visualizer_service import SongVisualizerService
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt

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
            max_duration=400
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
            "global_features": global_features,
            "success": True,
        }
    except Exception as e:
        st.error(f"Error processing audio: {e} for: {file.name}")
        return {"success": False, "error": str(e)}


# START OF UPLOAD
uploads_dir = Path("data/uploads")
uploads_dir.mkdir(exist_ok=True)

st.title("AI Music Mentor Dashboard")
st.markdown(
    "#### Upload your unfinished track to get helpful advice on how to finish it."
)
st.caption(
    "Our AI system is built from data of one experienced producers' feedback and gives advice in their subjective tone and style."
)


track_genre = st.selectbox("Unfinished Track Genre:", GENRES)
input_file = st.file_uploader(
    "Upload Unfinished track - MP3, WAV, or AIF file", type=["mp3", "wav", "aif"]
)

if input_file:
    st.audio(input_file)

ref_file = st.file_uploader(
    "Upload Reference track, something your aiming to get closer to - MP3, WAV, or AIF file",
    type=["mp3", "wav", "aif"],
)

if ref_file:
    st.audio(ref_file)

text_input = st.text_input("What do you need help with on your track?:")
dropdown_option = st.selectbox(
    "Stage your track is at:", ["Sketch", "Half Finished", "Almost Finished"]
)


if st.button("Submit"):
    if input_file and ref_file is not None and text_input:
        with st.spinner("🎵 Processing your tracks..."):
            # Create a session-specific folder
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_id = f"session_{timestamp}"
            session_dir = uploads_dir / session_id
            session_dir.mkdir(exist_ok=True)

            progress_bar = st.progress(0)
            status_text = st.empty()
            tip_container = st.empty()

            # Music production tips to show during loading
            music_tips = [
                "💡 **Tip**: Line up a reference track next to your track in your DAW to help you understand arrangement flow while your make your song. ",
                "🎵 **Did you know?**: The longer you spent on a track without breaks the more likely you are to lose touch with how it sounds? Don't forget to give your ears and mind a rest when creating!",
                "🔊 **Pro tip**: Make sure your kick and bass fundemntal frequency is at least 50Hz apart to avoid large overlap that ruin the power and clarity in your low end.",
                "🎛️ **Technique**: Side-chain compression can create that classic pumping effect, and when done subtly is also great to give space for other elements in the mix.",
                "⚡ **Energy tip**: Build tension with low volume subtle rising elements before your main drops can help lead the listener making impact more rewarding.",
                "🎶 **Arrangement**: Leave space in your mix - not every element needs to play at once, its often best to let one or two elements show off in each section.",
                "🔄 **Variation**: If you change multiple elements in A section at once it can breathe a whole new life into your tracks progression and is more impactful that just adding in hi hats or a drum variation.",
                "🎯 **Focus**: A great track usually has one main element that everything else supports, try to understand which element is the strongest and use that as your songs backbone.",
            ]

            import random

            current_tip = random.choice(music_tips)
            tip_container.info(current_tip)

            status_text.text("🎵 Creating session folder...")
            progress_bar.progress(10)

            status_text.text("🎧 Processing input track...")
            progress_bar.progress(20)
            # Show a new tip during processing
            current_tip = random.choice(music_tips)
            tip_container.info(current_tip)

            input_data = process_and_save_file(
                input_file,
                "input",
                session_dir,
                session_id,
                dropdown_option,
                text_input,
            )

            status_text.text("🎼 Processing reference track...")
            progress_bar.progress(40)
            # Show another tip
            current_tip = random.choice(music_tips)
            tip_container.info(current_tip)

            ref_data = process_and_save_file(
                ref_file,
                "reference",
                session_dir,
                session_id,
                dropdown_option,
                text_input,
            )

            if input_data["success"] and ref_data["success"]:
                status_text.text("💾 Saving to database...")
                progress_bar.progress(60)
                # Show final tip
                current_tip = random.choice(music_tips)
                tip_container.info(current_tip)

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
                    status_text.text("🎨 Generating arrangement analysis...")
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
                            from db.models import UserUpload

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

                                # Pre-generate visualizations
                                visualizer = SongVisualizerService()

                                # Input track visualization
                                if input_track_data and input_track_data.get(
                                    "raw_arrangement_pattern"
                                ):
                                    if input_track_data.get(
                                        "raw_predictions"
                                    ) and input_track_data.get("raw_confidence_scores"):
                                        import json
                                        import numpy as np
                                        from src.classifier.arrangement_postprocessing import (
                                            process_arrangement_predictions,
                                        )

                                        raw_predictions = json.loads(
                                            input_track_data["raw_predictions"]
                                        )
                                        confidence_scores = json.loads(
                                            input_track_data["raw_confidence_scores"]
                                        )

                                        blocks, analysis = (
                                            process_arrangement_predictions(
                                                np.array(raw_predictions),
                                                np.array(confidence_scores),
                                                ["O", "A", "B", "C"],
                                                min_segment_length=2,
                                                confidence_threshold=0.4,
                                            )
                                        )

                                        input_viz_fig = visualizer.plot_arrangement_waveform(
                                            audio_path=input_track_data["file_path"],
                                            arrangement_blocks=blocks,
                                            title=f"Input Track: {input_data['original_filename']}",
                                        )

                                # Reference track visualization
                                if ref_track_data and ref_track_data.get(
                                    "raw_arrangement_pattern"
                                ):
                                    if ref_track_data.get(
                                        "raw_predictions"
                                    ) and ref_track_data.get("raw_confidence_scores"):
                                        import json
                                        import numpy as np
                                        from src.classifier.arrangement_postprocessing import (
                                            process_arrangement_predictions,
                                        )

                                        raw_predictions = json.loads(
                                            ref_track_data["raw_predictions"]
                                        )
                                        confidence_scores = json.loads(
                                            ref_track_data["raw_confidence_scores"]
                                        )

                                        blocks, analysis = (
                                            process_arrangement_predictions(
                                                np.array(raw_predictions),
                                                np.array(confidence_scores),
                                                ["O", "A", "B", "C"],
                                                min_segment_length=2,
                                                confidence_threshold=0.4,
                                            )
                                        )

                                        ref_viz_fig = visualizer.plot_arrangement_waveform(
                                            audio_path=ref_track_data["file_path"],
                                            arrangement_blocks=blocks,
                                            title=f"Reference Track: {ref_data['original_filename']}",
                                        )

                        finally:
                            session.close()

                    except Exception as e:
                        arrangement_error = str(e)

                    # Generate AI feedback
                    status_text.text("🤖 Generating AI feedback...")
                    progress_bar.progress(95)

                    feedback = None
                    feedback_error = None
                    try:
                        # Create RAG service using existing database connection
                        rag_service = AudioRAG(db_ops.db)
                        feedback = rag_service.generate_feedback(
                            user_upload_id=upload_id,
                            question=text_input,
                            k=3,  # Get top 3 similar examples
                        )
                    except Exception as e:
                        feedback_error = str(e)

                    # Clear progress indicators
                    progress_bar.empty()
                    status_text.empty()
                    tip_container.empty()

                    # NOW DISPLAY EVERYTHING AT ONCE
                    st.success(f"✅ Successfully processed! Upload ID: {upload_id}")

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

                    # Display arrangement analysis
                    st.subheader("🎼 Arrangement Analysis")

                    if arrangement_error:
                        st.error(
                            f"❌ Could not load arrangement visualizations: {arrangement_error}"
                        )
                    else:
                        # Input track analysis
                        if input_track_data and input_track_data.get(
                            "raw_arrangement_pattern"
                        ):
                            st.markdown("### Input Track Analysis")
                            st.markdown(
                                f"**Arrangement Pattern:** `{input_track_data['smoothed_arrangement_pattern']}`"
                            )
                            if input_viz_fig:
                                st.pyplot(input_viz_fig)
                                plt.close(input_viz_fig)
                            if input_file:
                                st.audio(input_file)
                        else:
                            st.info("Input track arrangement analysis not available")

                        # Reference track analysis
                        if ref_track_data and ref_track_data.get(
                            "raw_arrangement_pattern"
                        ):
                            st.markdown("### Reference Track Analysis")
                            st.markdown(
                                f"**Arrangement Pattern:** `{ref_track_data['smoothed_arrangement_pattern']}`"
                            )
                            if ref_viz_fig:
                                st.pyplot(ref_viz_fig)
                                plt.close(ref_viz_fig)
                            if ref_file:
                                st.audio(ref_file)
                        else:
                            st.info(
                                "Reference track arrangement analysis not available"
                            )

                    # Display AI feedback
                    st.subheader("🎵 AI Music Mentor Feedback")
                    if feedback_error:
                        st.error(f"❌ Could not generate feedback: {feedback_error}")
                        st.info(
                            "💡 This might be because there are no training examples in the database yet."
                        )
                    else:
                        st.markdown(feedback)

                except Exception as e:
                    # Clear progress indicators on error
                    progress_bar.empty()
                    status_text.empty()
                    tip_container.empty()
                    st.error(f"❌ Database error: {e}")

            else:
                # Clear progress indicators on error
                progress_bar.empty()
                status_text.empty()
                tip_container.empty()
                st.error("Failed to process one or both audio files")

    else:
        if not input_file:
            st.warning("Please upload an input track")
        if not ref_file:
            st.warning("Please upload a reference track")
        if not text_input:
            st.warning("Please enter what you need help with")

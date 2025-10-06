import streamlit as st
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime
from src.audio_features import AudioFeatureService
from db.db import AudioRAGDatabase
from db.operations import AudioRAGOperations
from db.models import Track, TrainingExample
from dotenv import load_dotenv
import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd

load_dotenv()


@st.cache_resource
def get_database():
    """Initialize and return database connection"""
    connection_url = os.getenv(
        "DB_CONNECTION_URL", "postgresql://postgres:<ADD_TOENV_FILE>"
    )
    db = AudioRAGDatabase(connection_url)
    return AudioRAGOperations(db), db


def show_similarity_viz_tab():
    """Show the Similarity Visualization tab content"""
    st.markdown("#### 3D Similarity Visualization")
    st.caption(
        "Upload a track to see how it relates to all tracks in your database in 3D embedding space"
    )

    # File upload
    st.subheader("Upload Query Track")
    uploaded_file = st.file_uploader(
        "Upload track to analyze similarity",
        type=["mp3", "wav", "aif", "aiff"],
        key="similarity_query",
    )

    if uploaded_file:
        st.audio(uploaded_file)

        col1, col2 = st.columns(2)
        with col1:
            k_similar = st.slider("Number of similar tracks to highlight", 3, 10, 5)
            similarity_metric = st.selectbox(
                "Similarity Metric", ["cosine", "euclidean", "inner_product"]
            )
        with col2:
            reduction_method = st.selectbox(
                "Dimensionality Reduction", ["PCA", "t-SNE"]
            )
            training_only = st.checkbox(
                "Only search INPUT tracks from training examples",
                value=False,
                help="Only searches tracks that are input tracks in training examples (excludes reference tracks)",
            )

        if st.button("Generate 3D Similarity Visualization", type="primary"):
            with st.spinner("Processing audio and generating visualization..."):
                try:
                    # Process uploaded audio
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    # Keep original file extension for better audio processing
                    file_extension = Path(uploaded_file.name).suffix.lower()
                    if file_extension not in [".mp3", ".wav", ".aif", ".aiff"]:
                        file_extension = ".mp3"  # Default fallback
                    temp_path = f"temp_query_{timestamp}{file_extension}"

                    # Save temporary file
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    # Extract features
                    service = AudioFeatureService()
                    global_features = service.load_audio_file(
                        temp_path
                    ).extract_global_features(max_duration=400)
                    query_embedding = service.create_embedding_vector(global_features)

                    # Clean up temp file
                    os.remove(temp_path)

                    # Get database
                    db_ops, db = get_database()
                    session = db.get_session()

                    try:
                        # ALWAYS get ALL tracks for visualization (to show complete space)
                        all_tracks = (
                            session.query(Track)
                            .filter(Track.global_embedding.isnot(None))
                            .all()
                        )

                        # But for similarity search, get filtered set if needed
                        if training_only:
                            # Get only tracks that are INPUT tracks in training examples
                            search_tracks_query = (
                                session.query(Track)
                                .filter(Track.global_embedding.isnot(None))
                                .join(
                                    TrainingExample,
                                    TrainingExample.example_track_id == Track.id,
                                )
                                .all()
                            )
                            # Remove duplicates manually since DISTINCT doesn't work with JSON columns
                            seen_ids = set()
                            search_tracks = []
                            for track in search_tracks_query:
                                if track.id not in seen_ids:
                                    search_tracks.append(track)
                                    seen_ids.add(track.id)

                            if len(search_tracks) < 1:
                                st.error(
                                    f"No tracks with training examples found for similarity search. Found: {len(search_tracks)}"
                                )
                                return
                        else:
                            search_tracks = all_tracks

                        if len(all_tracks) < 3:
                            st.error(
                                f"Need at least 3 tracks with embeddings in database for visualization. Found: {len(all_tracks)}"
                            )
                            return

                        # Prepare data - use ALL tracks for visualization
                        embeddings = []
                        track_info = []

                        # Add all database tracks (for complete visualization)
                        for track in all_tracks:
                            embeddings.append(list(track.global_embedding))
                            track_info.append(
                                {
                                    "name": Path(str(track.file_path)).stem,
                                    "id": track.id,
                                    "type": "database",
                                    "duration": track.duration,
                                }
                            )

                        # Add query track
                        embeddings.append(query_embedding)
                        track_info.append(
                            {
                                "name": f"Query: {uploaded_file.name}",
                                "id": -1,
                                "type": "query",
                                "duration": len(global_features.get("tempo", [0]))
                                * 0.1,  # rough estimate
                            }
                        )

                        embeddings_array = np.array(embeddings)

                        # Choose which method to use based on filter
                        if training_only:
                            similar_tracks = (
                                db_ops.find_similar_tracks_with_training_examples(
                                    embedding=list(query_embedding),
                                    metric=similarity_metric,
                                    limit=k_similar,
                                )
                            )
                            search_method = "find_similar_tracks_with_training_examples"
                        else:
                            similar_tracks = db_ops.find_similar_tracks(
                                embedding=list(query_embedding),
                                metric=similarity_metric,
                                limit=k_similar,
                            )
                            search_method = "find_similar_tracks"

                        # Now get the actual distance/similarity scores for these tracks
                        # We need to run the same query but capture the distance values
                        session_for_distances = db.get_session()
                        try:
                            # Build base query with distance/score calculation
                            base_query = None
                            order_by = None

                            if similarity_metric == "cosine":
                                distance_col = Track.global_embedding.cosine_distance(
                                    list(query_embedding)
                                )
                                base_query = session_for_distances.query(
                                    Track, distance_col.label("distance")
                                )
                                order_by = distance_col
                            elif similarity_metric == "euclidean":
                                distance_col = Track.global_embedding.l2_distance(
                                    list(query_embedding)
                                )
                                base_query = session_for_distances.query(
                                    Track, distance_col.label("distance")
                                )
                                order_by = distance_col
                            elif similarity_metric == "inner_product":
                                score_col = Track.global_embedding.max_inner_product(
                                    list(query_embedding)
                                )
                                base_query = session_for_distances.query(
                                    Track, score_col.label("distance")
                                )
                                order_by = score_col.desc()

                            # Add training filter if needed (only INPUT tracks)
                            if training_only and base_query is not None:
                                base_query = base_query.join(
                                    TrainingExample,
                                    TrainingExample.example_track_id == Track.id,
                                )

                            # Execute query and handle duplicates
                            if base_query is not None and order_by is not None:
                                results_with_distances = (
                                    base_query.order_by(order_by)
                                    .limit(k_similar * 2)
                                    .all()
                                )  # Get more to handle duplicates

                                # Remove duplicates manually and limit results
                                seen_ids = set()
                                unique_results = []
                                for row in results_with_distances:
                                    if (
                                        row[0].id not in seen_ids
                                        and len(unique_results) < k_similar
                                    ):
                                        unique_results.append(row)
                                        seen_ids.add(row[0].id)

                                # Extract tracks and their actual distance/similarity values
                                similar_tracks_with_scores = [
                                    (row[0], float(row[1])) for row in unique_results
                                ]
                            else:
                                similar_tracks_with_scores = [
                                    (track, 0.0) for track in similar_tracks[:k_similar]
                                ]

                        finally:
                            session_for_distances.close()

                        # Create mapping of similar track IDs to their ranking and scores
                        similar_track_data = {}
                        for i, (track, score) in enumerate(similar_tracks_with_scores):
                            similar_track_data[track.id] = {"rank": i, "score": score}

                        # Get top similar track indices in our ALL_TRACKS array (for visualization)
                        top_indices = []
                        for i, track in enumerate(all_tracks):
                            if track.id in similar_track_data:
                                top_indices.append(i)

                        # Calculate display similarities for visualization
                        similarities = []
                        for track in all_tracks:
                            if track.id in similar_track_data:
                                rank = similar_track_data[track.id]["rank"]
                                # For display - approximate similarity (1 = most similar, decreasing)
                                similarities.append(1.0 - (rank / k_similar) * 0.5)
                            else:
                                similarities.append(
                                    0.1
                                )  # Low similarity for non-matches
                        similarities = np.array(similarities)

                        # Reduce dimensions
                        if reduction_method == "PCA":
                            reducer = PCA(n_components=3, random_state=42)
                        else:
                            reducer = TSNE(
                                n_components=3,
                                random_state=42,
                                perplexity=min(30, len(embeddings) // 2),
                            )

                        coords_3d = reducer.fit_transform(embeddings_array)

                        # Prepare visualization data with better colors and sizes
                        viz_data = []
                        for i, (coord, info) in enumerate(zip(coords_3d, track_info)):
                            if info["type"] == "query":
                                color = "rgba(255, 0, 0, 0.9)"  # Bright red for query
                                size = 20
                                symbol = "diamond"
                            elif i in top_indices:
                                # Gradient from green to yellow based on similarity rank
                                rank_in_similar = top_indices.index(i)
                                if rank_in_similar == 0:
                                    color = (
                                        "rgba(0, 255, 0, 0.9)"  # Bright green for #1
                                    )
                                elif rank_in_similar < 3:
                                    color = "rgba(50, 205, 50, 0.8)"  # Medium green for top 3
                                else:
                                    color = "rgba(255, 215, 0, 0.8)"  # Gold for others
                                size = 15 - rank_in_similar  # Larger for more similar
                                symbol = "circle"
                            else:
                                color = "rgba(173, 216, 230, 0.4)"  # Light blue, more transparent
                                size = 5
                                symbol = "circle"

                            viz_data.append(
                                {
                                    "x": coord[0],
                                    "y": coord[1],
                                    "z": coord[2],
                                    "name": info["name"],
                                    "color": color,
                                    "size": size,
                                    "symbol": symbol,
                                    "similarity": (
                                        similarities[i]
                                        if i < len(similarities)
                                        else 0.1
                                    ),
                                    "type": info["type"],
                                }
                            )

                        # Create separate traces for better legend and visibility
                        df = pd.DataFrame(viz_data)

                        fig = go.Figure()

                        # Add query track
                        query_data = df[df["type"] == "query"]
                        if not query_data.empty:
                            fig.add_trace(
                                go.Scatter3d(
                                    x=query_data["x"],
                                    y=query_data["y"],
                                    z=query_data["z"],
                                    mode="markers",
                                    marker=dict(
                                        size=20,
                                        color="red",
                                        symbol="diamond",
                                        line=dict(width=2, color="darkred"),
                                    ),
                                    text=query_data["name"],
                                    name="Your Query Track",
                                    hovertemplate="<b>%{text}</b><br>🎯 Query Track<extra></extra>",
                                    showlegend=True,
                                )
                            )

                        # Add similar tracks
                        similar_data = df[df.index.isin(top_indices)]
                        if not similar_data.empty:
                            fig.add_trace(
                                go.Scatter3d(
                                    x=similar_data["x"],
                                    y=similar_data["y"],
                                    z=similar_data["z"],
                                    mode="markers",
                                    marker=dict(
                                        size=similar_data["size"],
                                        color="green",
                                        symbol="circle",
                                        line=dict(width=1, color="darkgreen"),
                                    ),
                                    text=similar_data["name"],
                                    name=f"Top {k_similar} Similar",
                                    hovertemplate="<b>%{text}</b><br>✅ Similar Track<extra></extra>",
                                    showlegend=True,
                                )
                            )

                        # Add other tracks
                        other_indices = [
                            i for i in range(len(df) - 1) if i not in top_indices
                        ]  # -1 to exclude query
                        other_data = df[df.index.isin(other_indices)]
                        if not other_data.empty:
                            fig.add_trace(
                                go.Scatter3d(
                                    x=other_data["x"],
                                    y=other_data["y"],
                                    z=other_data["z"],
                                    mode="markers",
                                    marker=dict(
                                        size=5,
                                        color="lightblue",
                                        opacity=0.4,
                                        symbol="circle",
                                    ),
                                    text=other_data["name"],
                                    name="Other Tracks",
                                    hovertemplate="<b>%{text}</b><br>📁 Database Track<extra></extra>",
                                    showlegend=True,
                                )
                            )

                        # Explanation of what the axes mean
                        axis_explanation = {
                            "PCA": "Each axis represents a principal component - directions of maximum variance in your audio embeddings. Tracks that sound similar cluster together.",
                            "t-SNE": "Each axis represents a dimension optimized to show local similarity relationships. Similar tracks form tight clusters, different tracks are pushed apart.",
                        }

                        # Update title based on search method
                        title_text = (
                            f"🎵 3D Audio Similarity Space ({similarity_metric} metric)"
                        )
                        if training_only:
                            title_text += " - Input Training Tracks Only"
                        title_text += (
                            f"<br><sup>{axis_explanation[reduction_method]}</sup>"
                        )

                        fig.update_layout(
                            title=dict(text=title_text, x=0.5, font=dict(size=16)),
                            scene=dict(
                                xaxis_title=f"{reduction_method} Component 1<br><sub>Audio Feature Dimension</sub>",
                                yaxis_title=f"{reduction_method} Component 2<br><sub>Audio Feature Dimension</sub>",
                                zaxis_title=f"{reduction_method} Component 3<br><sub>Audio Feature Dimension</sub>",
                                camera=dict(
                                    eye=dict(
                                        x=1.5, y=1.5, z=1.5
                                    )  # Better default viewing angle
                                ),
                                bgcolor="rgba(240,240,240,0.1)",
                                xaxis=dict(
                                    showgrid=True, gridcolor="lightgray", gridwidth=1
                                ),
                                yaxis=dict(
                                    showgrid=True, gridcolor="lightgray", gridwidth=1
                                ),
                                zaxis=dict(
                                    showgrid=True, gridcolor="lightgray", gridwidth=1
                                ),
                            ),
                            height=700,  # Taller
                            margin=dict(l=0, r=0, b=50, t=80),
                            legend=dict(
                                yanchor="top",
                                y=0.99,
                                xanchor="left",
                                x=0.01,
                                bgcolor="rgba(255,255,255,0.8)",
                            ),
                        )

                        st.plotly_chart(fig, use_container_width=True)

                        # Show similarity results using the actual returned tracks with scores
                        method_display = (
                            "Input Training Tracks Only" if training_only else "All Tracks"
                        )
                        st.subheader(
                            f"Top Similar Tracks ({method_display} - {similarity_metric} metric)"
                        )
                        st.caption(f"Using method: `{search_method}`")

                        similar_tracks_info = []
                        for i, (track, score) in enumerate(similar_tracks_with_scores):
                            # Format the score based on metric type
                            if similarity_metric == "inner_product":
                                score_display = f"{score:.4f}"
                                score_desc = "Higher = More Similar"
                            else:  # cosine or euclidean distance
                                score_display = f"{score:.4f}"
                                score_desc = "Lower = More Similar"

                            similar_tracks_info.append(
                                {
                                    "Rank": i + 1,
                                    "Track": Path(str(track.file_path)).stem,
                                    f"{similarity_metric.title()} Score": score_display,
                                    "Duration": f"{track.duration:.1f}s",
                                    "ID": track.id,
                                }
                            )

                        df_results = pd.DataFrame(similar_tracks_info)
                        st.dataframe(df_results, use_container_width=True)

                        # Add explanation of what the scores mean
                        if similarity_metric == "cosine":
                            st.caption(
                                "📊 **Cosine Distance**: 0.0 = identical, 2.0 = completely opposite. Lower values = more similar."
                            )
                        elif similarity_metric == "euclidean":
                            st.caption(
                                "📊 **Euclidean Distance**: 0.0 = identical, higher values = more different. Lower values = more similar."
                            )
                        elif similarity_metric == "inner_product":
                            st.caption(
                                "📊 **Inner Product**: Higher positive values = more similar. Measures alignment between embeddings."
                            )

                        # Legend
                        st.markdown(
                            """
                        **Legend:**
                        - 🔴 **Red**: Your uploaded query track
                        - 🟢 **Green**: Top similar tracks
                        - 🔵 **Light Blue**: Other database tracks
                        """
                        )

                    finally:
                        session.close()

                except Exception as e:
                    st.error(f"Error: {e}")
                    import traceback

                    st.code(traceback.format_exc())
    else:
        st.info("Upload an MP3 file to begin similarity analysis")

    # Database stats
    st.markdown("---")
    st.subheader("Database Info")

    try:
        db_ops, db = get_database()
        session = db.get_session()

        try:
            total_tracks = session.query(Track).count()
            tracks_with_embeddings = (
                session.query(Track).filter(Track.global_embedding.isnot(None)).count()
            )

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Tracks", total_tracks)
            with col2:
                st.metric("Tracks with Embeddings", tracks_with_embeddings)

        finally:
            session.close()
    except Exception as e:
        st.error(f"Database connection error: {e}")

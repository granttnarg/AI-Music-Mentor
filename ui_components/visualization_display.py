import streamlit as st
import matplotlib.pyplot as plt
from typing import Optional, Any


def render_arrangement_visualization(
    arrangement_error: Optional[str],
    comparison_viz_fig: Optional[Any],
    input_viz_fig: Optional[Any],
    ref_viz_fig: Optional[Any],
    input_track_data: Optional[dict],
    ref_track_data: Optional[dict],
    input_file: Optional[Any],
    ref_file: Optional[Any],
):
    """
    Render arrangement analysis visualizations.

    Args:
        arrangement_error: Error message if arrangement loading failed
        comparison_viz_fig: Combined comparison visualization figure
        input_viz_fig: Input track visualization figure
        ref_viz_fig: Reference track visualization figure
        input_track_data: Input track metadata and features
        ref_track_data: Reference track metadata and features
        input_file: Input audio file for playback
        ref_file: Reference audio file for playback
    """
    st.markdown("**Arrangement Analysis**")

    if arrangement_error:
        st.error(f"Could not load arrangement visualizations: {arrangement_error}")
    else:
        # Time-aligned comparison visualization
        if comparison_viz_fig is not None:
            _render_comparison_view(
                comparison_viz_fig,
                input_track_data,
                ref_track_data,
                input_file,
                ref_file,
            )
        else:
            _render_individual_view(
                input_viz_fig,
                ref_viz_fig,
                input_track_data,
                ref_track_data,
                input_file,
                ref_file,
            )


def _get_tempo_string(track_data: Optional[dict]) -> str:
    """Extract tempo string from track data."""
    if not track_data or not track_data.get("global_feature_data"):
        return ""

    if not isinstance(track_data["global_feature_data"], dict):
        return ""

    tempo = track_data["global_feature_data"].get("rhythm", {}).get("tempo")
    return f" ({tempo:.0f} BPM)" if tempo else ""


def _render_comparison_view(
    comparison_viz_fig: Any,
    input_track_data: Optional[dict],
    ref_track_data: Optional[dict],
    input_file: Optional[Any],
    ref_file: Optional[Any],
):
    """Render time-aligned comparison view."""
    # Show patterns with BPM in compact format
    pattern_col1, pattern_col2 = st.columns(2)

    with pattern_col1:
        if input_track_data and input_track_data.get("smoothed_arrangement_pattern"):
            input_tempo = _get_tempo_string(input_track_data)
            st.markdown(
                f"**Input:** `{input_track_data['smoothed_arrangement_pattern']}`{input_tempo}"
            )

    with pattern_col2:
        if ref_track_data and ref_track_data.get("smoothed_arrangement_pattern"):
            ref_tempo = _get_tempo_string(ref_track_data)
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


def _render_individual_view(
    input_viz_fig: Optional[Any],
    ref_viz_fig: Optional[Any],
    input_track_data: Optional[dict],
    ref_track_data: Optional[dict],
    input_file: Optional[Any],
    ref_file: Optional[Any],
):
    """Render individual track visualizations side by side."""
    # Show patterns first
    pattern_col1, pattern_col2 = st.columns(2)

    with pattern_col1:
        if input_track_data and input_track_data.get("raw_arrangement_pattern"):
            input_tempo = _get_tempo_string(input_track_data)
            st.markdown(
                f"**Input:** `{input_track_data['smoothed_arrangement_pattern']}`{input_tempo}"
            )
        else:
            st.info("Input pattern not available")

    with pattern_col2:
        if ref_track_data and ref_track_data.get("raw_arrangement_pattern"):
            ref_tempo = _get_tempo_string(ref_track_data)
            st.markdown(
                f"**Reference:** `{ref_track_data['smoothed_arrangement_pattern']}`{ref_tempo}"
            )
        else:
            st.info("Reference pattern not available")

    # Individual visualizations
    viz_col1, viz_col2 = st.columns(2)

    with viz_col1:
        if input_viz_fig is not None:
            st.pyplot(input_viz_fig, use_container_width=True)
            plt.close(input_viz_fig)
        st.markdown("**Input Track Audio:**")
        if input_file:
            st.audio(input_file, format="audio/mp3")

    with viz_col2:
        if ref_viz_fig is not None:
            st.pyplot(ref_viz_fig, use_container_width=True)
            plt.close(ref_viz_fig)
        st.markdown("**Reference Track Audio:**")
        if ref_file:
            st.audio(ref_file, format="audio/mp3")

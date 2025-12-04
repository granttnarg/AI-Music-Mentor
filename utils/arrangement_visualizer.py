"""
Utility functions for preparing arrangement visualizations.

Pure functions that process track data and generate visualization figures
without touching the database directly.
"""

import json
import numpy as np
from typing import Dict, Optional, Any
from services.song_visualizer_service import SongVisualizerService
from src.classifier.arrangement_postprocessing import process_arrangement_predictions


def prepare_arrangement_visualizations(
    input_track_data: Optional[Dict],
    ref_track_data: Optional[Dict],
    input_filename: str,
    ref_filename: str,
) -> Dict[str, Any]:
    """
    Prepare arrangement visualizations for input and reference tracks.

    Args:
        input_track_data: Input track data from database (contains predictions, file_path, etc.)
        ref_track_data: Reference track data from database (contains predictions, file_path, etc.)
        input_filename: Original filename for input track (for display)
        ref_filename: Original filename for reference track (for display)

    Returns:
        dict with keys:
            - comparison_viz_fig: Combined comparison figure (if both have data)
            - input_viz_fig: Individual input figure (fallback)
            - ref_viz_fig: Individual reference figure (fallback)
            - error: Error message if any
    """
    result = {
        "comparison_viz_fig": None,
        "input_viz_fig": None,
        "ref_viz_fig": None,
        "error": None,
    }

    try:
        visualizer = SongVisualizerService()
        input_blocks = None
        ref_blocks = None

        # Process input track
        if input_track_data and input_track_data.get("raw_arrangement_pattern"):
            if input_track_data.get("raw_predictions") and input_track_data.get(
                "raw_confidence_scores"
            ):
                raw_predictions = json.loads(input_track_data["raw_predictions"])
                confidence_scores = json.loads(
                    input_track_data["raw_confidence_scores"]
                )

                input_blocks, _ = process_arrangement_predictions(
                    np.array(raw_predictions),
                    np.array(confidence_scores),
                    ["O", "A", "B", "C"],
                    min_segment_length=2,
                    confidence_threshold=0.4,
                )

        # Process reference track
        if ref_track_data and ref_track_data.get("raw_arrangement_pattern"):
            if ref_track_data.get("raw_predictions") and ref_track_data.get(
                "raw_confidence_scores"
            ):
                raw_predictions = json.loads(ref_track_data["raw_predictions"])
                confidence_scores = json.loads(ref_track_data["raw_confidence_scores"])

                ref_blocks, _ = process_arrangement_predictions(
                    np.array(raw_predictions),
                    np.array(confidence_scores),
                    ["O", "A", "B", "C"],
                    min_segment_length=2,
                    confidence_threshold=0.4,
                )

        # Create time-aligned comparison if both tracks have arrangement data
        if input_blocks and ref_blocks:
            result["comparison_viz_fig"] = visualizer.plot_time_aligned_comparison(
                input_audio_path=input_track_data["file_path"],
                input_arrangement_blocks=input_blocks,
                input_title=f"Input Track: {input_filename}",
                reference_audio_path=ref_track_data["file_path"],
                reference_arrangement_blocks=ref_blocks,
                reference_title=f"Reference Track: {ref_filename}",
            )
        else:
            # Fallback to individual plots if one track is missing arrangement data
            if input_blocks:
                result["input_viz_fig"] = visualizer.plot_arrangement_waveform(
                    audio_path=input_track_data["file_path"],
                    arrangement_blocks=input_blocks,
                    title=f"Input Track: {input_filename}",
                )
            if ref_blocks:
                result["ref_viz_fig"] = visualizer.plot_arrangement_waveform(
                    audio_path=ref_track_data["file_path"],
                    arrangement_blocks=ref_blocks,
                    title=f"Reference Track: {ref_filename}",
                )

    except Exception as e:
        result["error"] = str(e)

    return result

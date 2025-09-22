#!/usr/bin/env python3
"""
Post-processing utilities for arrangement classification to create cleaner segment transitions.
"""

import numpy as np
from typing import List, Dict

# min seg 2 and min confidence 0.4 gave the best naunced by smoother output so far.
def smooth_predictions(predictions: np.ndarray,
                      confidence_scores: np.ndarray,
                      min_segment_length: int = 2,
                      confidence_threshold: float = 0.4) -> np.ndarray:
    """
    Smooth arrangement predictions to remove short segments and create cleaner transitions.

    Args:
        predictions: Array of predicted class indices
        confidence_scores: Array of confidence scores for each prediction
        min_segment_length: Minimum number of consecutive segments for a class
        confidence_threshold: Minimum confidence to trust a prediction

    Returns:
        Smoothed predictions array
    """
    smoothed = predictions.copy()
    n_segments = len(predictions)

    # Step 1: Filter low-confidence predictions by replacing with most confident neighbor
    for i in range(n_segments):
        if confidence_scores[i] < confidence_threshold:
            # Look at neighbors to find most confident replacement
            neighbors = []
            if i > 0:
                neighbors.append((i-1, confidence_scores[i-1]))
            if i < n_segments - 1:
                neighbors.append((i+1, confidence_scores[i+1]))

            if neighbors:
                # Use the class from the most confident neighbor
                best_neighbor_idx = max(neighbors, key=lambda x: x[1])[0]
                smoothed[i] = predictions[best_neighbor_idx]

    # Step 2: Remove short segments
    i = 0
    while i < n_segments:
        current_class = smoothed[i]
        segment_start = i

        # Find the end of this segment
        while i < n_segments and smoothed[i] == current_class:
            i += 1
        segment_end = i
        segment_length = segment_end - segment_start

        # If segment is too short, merge with neighbors
        if segment_length < min_segment_length:
            # Decide what to replace it with
            replacement_class = None

            # Look at surrounding context
            before_class = smoothed[segment_start - 1] if segment_start > 0 else None
            after_class = smoothed[segment_end] if segment_end < n_segments else None

            if before_class is not None and after_class is not None:
                if before_class == after_class:
                    # Surrounded by same class - use that
                    replacement_class = before_class
                else:
                    # Different classes - use the one with higher average confidence
                    before_conf = np.mean(confidence_scores[max(0, segment_start-2):segment_start])
                    after_conf = np.mean(confidence_scores[segment_end:min(n_segments, segment_end+2)])
                    replacement_class = before_class if before_conf > after_conf else after_class
            elif before_class is not None:
                replacement_class = before_class
            elif after_class is not None:
                replacement_class = after_class
            else:
                # Fallback to most common class in the track
                replacement_class = np.bincount(smoothed).argmax()

            # Apply the replacement
            if replacement_class is not None:
                smoothed[segment_start:segment_end] = replacement_class

        # Move to next segment (but don't increment if we just merged)
        if segment_length >= min_segment_length:
            continue
        else:
            i = segment_start + 1  # Start over from the merged area

    return smoothed

def create_arrangement_blocks(predictions: np.ndarray,
                            class_names: List[str],
                            segment_duration: float = 5.0) -> List[Dict]:
    """
    Convert predictions into arrangement blocks with start/end times.

    Args:
        predictions: Prediction array (can be raw or smoothed)
        class_names: List of class names ['O', 'A', 'B', 'C']
        segment_duration: Duration of each segment in seconds

    Returns:
        List of arrangement blocks with metadata
    """
    blocks = []

    if len(predictions) == 0:
        return blocks

    current_class = predictions[0]
    block_start_idx = 0

    for i in range(1, len(predictions)):
        if predictions[i] != current_class:
            # End of current block
            block_end_idx = i

            start_time = block_start_idx * segment_duration
            end_time = block_end_idx * segment_duration
            duration = end_time - start_time

            block = {
                'start_time': start_time,
                'end_time': end_time,
                'duration': duration,
                'start_mm_ss': f"{int(start_time//60):02d}:{start_time%60:05.2f}",
                'end_mm_ss': f"{int(end_time//60):02d}:{end_time%60:05.2f}",
                'arrangement_section': class_names[current_class],
                'section_index': len(blocks),
                'segment_count': block_end_idx - block_start_idx
            }
            blocks.append(block)

            # Start new block
            current_class = predictions[i]
            block_start_idx = i

    # Handle the last block
    if block_start_idx < len(predictions):
        start_time = block_start_idx * segment_duration
        end_time = len(predictions) * segment_duration
        duration = end_time - start_time

        block = {
            'start_time': start_time,
            'end_time': end_time,
            'duration': duration,
            'start_mm_ss': f"{int(start_time//60):02d}:{start_time%60:05.2f}",
            'end_mm_ss': f"{int(end_time//60):02d}:{end_time%60:05.2f}",
            'arrangement_section': class_names[current_class],
            'section_index': len(blocks),
            'segment_count': len(predictions) - block_start_idx
        }
        blocks.append(block)

    return blocks

def suggest_arrangement_pattern(blocks: List[Dict], min_duration_threshold: float = 20.0) -> Dict:
    """
    Analyze arrangement blocks and create both detailed and simplified patterns.

    Args:
        blocks: List of arrangement blocks
        min_duration_threshold: Minimum duration for sections to be included in simplified pattern

    Returns:
        Dictionary with pattern analysis including both detailed and simplified patterns
    """
    if not blocks:
        return {}

    # Get sequence of sections with duration info
    section_sequence = [block['arrangement_section'] for block in blocks]

    # Create detailed pattern with * for short sections
    detailed_pattern_parts = []
    simplified_sections = []

    for block in blocks:
        section = block['arrangement_section']
        duration = block['duration']

        if duration < min_duration_threshold:
            # Mark short sections with *
            detailed_pattern_parts.append(f"*{section}")
        else:
            detailed_pattern_parts.append(section)
            simplified_sections.append(section)

    # Create pattern strings
    detailed_pattern = '-'.join(detailed_pattern_parts)
    simplified_pattern = '-'.join(simplified_sections) if simplified_sections else ""

    # Calculate section statistics
    section_stats = {}
    total_duration = sum(block['duration'] for block in blocks)

    for section in ['O', 'A', 'B', 'C']:
        section_blocks = [b for b in blocks if b['arrangement_section'] == section]
        if section_blocks:
            total_section_duration = sum(b['duration'] for b in section_blocks)
            section_stats[section] = {
                'count': len(section_blocks),
                'total_duration': total_section_duration,
                'percentage': (total_section_duration / total_duration) * 100,
                'avg_duration': total_section_duration / len(section_blocks),
                'durations': [b['duration'] for b in section_blocks],
                'short_sections': len([b for b in section_blocks if b['duration'] < min_duration_threshold])
            }
        else:
            section_stats[section] = {
                'count': 0,
                'total_duration': 0,
                'percentage': 0,
                'avg_duration': 0,
                'durations': [],
                'short_sections': 0
            }

    # Identify common patterns (using simplified pattern)
    patterns = []
    simplified_str = ''.join(simplified_sections)

    # Simple pattern detection
    if 'ABA' in simplified_str:
        patterns.append('ABA (High-Breakdown-High)')
    if 'ABC' in simplified_str:
        patterns.append('ABC (High-Breakdown-Low)')
    if 'OAO' in simplified_str:
        patterns.append('OAO (Build-Drop-Build)')

    # Classify overall structure (based on significant sections only)
    significant_stats = {k: v for k, v in section_stats.items() if v['total_duration'] >= min_duration_threshold}
    total_significant_duration = sum(stats['total_duration'] for stats in significant_stats.values())

    if total_significant_duration > 0:
        if significant_stats.get('B', {}).get('total_duration', 0) / total_significant_duration > 0.4:
            structure_type = 'High Energy Dominant'
        elif significant_stats.get('O', {}).get('total_duration', 0) / total_significant_duration > 0.6:
            structure_type = 'Steady Groove'
        elif significant_stats.get('C', {}).get('total_duration', 0) / total_significant_duration > 0.3:
            structure_type = 'Ambient/Breakdown Heavy'
        else:
            structure_type = 'Balanced Mix'
    else:
        structure_type = 'Fragmented'

    return {
        'section_sequence': section_sequence,
        'detailed_pattern': detailed_pattern,
        'simplified_pattern': simplified_pattern,
        'section_stats': section_stats,
        'detected_patterns': patterns,
        'structure_type': structure_type,
        'total_duration': total_duration,
        'total_blocks': len(blocks),
        'simplified_blocks': len(simplified_sections),
        'min_duration_threshold': min_duration_threshold
    }
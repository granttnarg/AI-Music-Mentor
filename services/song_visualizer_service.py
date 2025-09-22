"""
Song Visualizer Service

Provides visualization capabilities for arrangement classification results.
Shows audio waveforms with colored sections representing different arrangement parts.
"""

import os
import numpy as np
import librosa
import librosa.display
from matplotlib import pyplot as plt
from typing import List, Dict, Optional, Tuple
from pathlib import Path


class SongVisualizerService:
    """
    Service for visualizing arrangement classification results on audio waveforms.
    """

    def __init__(self):
        """Initialize the visualizer service."""
        # Color scheme for arrangement sections
        self.class_colors = {
            'O': 'lightgray',    # Intro/Outro/Other
            'A': 'blue',       # Medium Energy
            'B': 'red',          # High Energy (drops, climaxes)
            'C': 'yellow'          # Breakdown/transitions
        }

        # Y-positions for timeline plot
        self.class_y_positions = {
            'O': 1,    # Intro/Outro/Other (middle-low)
            'A': 2,    # Medium Energy (middle)
            'B': 3,    # High Energy (top)
            'C': 0     # Breakdown (bottom)
        }

    def plot_raw_predictions_waveform(self,
                                     audio_path: str,
                                     raw_predictions: np.ndarray,
                                     confidence_scores: np.ndarray,
                                     audio_features: any,
                                     class_names: List[str],
                                     title: Optional[str] = None,
                                     save_path: Optional[str] = None,
                                     figsize: Tuple[int, int] = (15, 6)) -> plt.Figure:
        """
        Plot audio waveform with RAW arrangement predictions overlaid (before smoothing).
        
        Args:
            audio_path: Path to audio file
            raw_predictions: Raw prediction array from model
            confidence_scores: Confidence scores for each prediction
            audio_features: Audio features object with meter grid
            class_names: List of class names ['O', 'A', 'B', 'C']
            title: Optional plot title
            save_path: Optional path to save the plot
            figsize: Figure size (width, height)
            
        Returns:
            matplotlib Figure object
        """
        # Load audio for visualization
        y, sr = librosa.load(audio_path, sr=None)
        y_harm, y_perc = librosa.effects.hpss(y)
        
        plt.close('all')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), dpi=96)
        
        # Top plot: Waveform with raw predictions
        librosa.display.waveshow(y_harm, sr=sr, alpha=0.8, ax=ax1, 
                                color='deepskyblue', label='Harmonic')
        librosa.display.waveshow(y_perc, sr=sr, alpha=0.7, ax=ax1, 
                                color='plum', label='Percussive')
        
        # Calculate meter times
        meter_grid_times = librosa.frames_to_time(
            audio_features.meter_grid, sr=audio_features.sr, hop_length=audio_features.hop_length)
        
        # Track which classes we've added to legend
        legend_added = set()
        
        # Highlight sections by raw predictions
        for i, prediction in enumerate(raw_predictions):
            if i < len(meter_grid_times) - 1:
                start_time = meter_grid_times[i]
                end_time = meter_grid_times[i + 1]
                section = class_names[prediction]
                confidence = confidence_scores[i]
                
                color = self.class_colors.get(section, 'black')
                
                # Add to legend only once per class
                label = f"{section} - {self._get_section_description(section)}" if section not in legend_added else None
                if label:
                    legend_added.add(section)
                
                # Use alpha based on confidence (lower confidence = more transparent)
                alpha = max(0.2, confidence * 0.6)  # Min 0.2, max based on confidence
                ax1.axvspan(start_time, end_time, color=color, alpha=alpha, label=label)
        
        # Configure top plot
        duration = len(y) / sr
        ax1.set_xlim([0, duration])
        ax1.set_ylabel('Amplitude')
        ax1.set_title(f'RAW Predictions (Before Smoothing): {Path(audio_path).stem}')
        ax1.legend(loc='upper right')
        
        # Bottom plot: Confidence scores over time
        segment_times = [(meter_grid_times[i] + meter_grid_times[i+1])/2 
                        for i in range(len(raw_predictions))]
        
        # Color code confidence by predicted class
        colors = [self.class_colors.get(class_names[pred], 'black') for pred in raw_predictions]
        ax2.scatter(segment_times, confidence_scores, c=colors, alpha=0.7, s=20)
        ax2.plot(segment_times, confidence_scores, color='gray', alpha=0.5, linewidth=1)
        
        ax2.set_xlim([0, duration])
        ax2.set_ylim([0, 1])
        ax2.set_ylabel('Confidence')
        ax2.set_xlabel('Time (mm:ss)')
        ax2.set_title('Prediction Confidence Over Time')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='50% confidence')
        ax2.legend()
        
        # Set time-based x-axis labels for both plots
        xticks = np.arange(0, duration, 30)  # Every 30 seconds
        xlabels = [f"{int(tick // 60)}:{int(tick % 60):02d}" for tick in xticks]
        
        ax1.set_xticks(xticks)
        ax1.set_xticklabels([])  # No labels on top plot
        
        ax2.set_xticks(xticks)
        ax2.set_xticklabels(xlabels)
        
        plt.tight_layout()
        
        # Save if path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Raw predictions visualization saved to {save_path}")
        
        return fig

    def plot_high_confidence_predictions_waveform(self,
                                                 audio_path: str,
                                                 raw_predictions: np.ndarray,
                                                 confidence_scores: np.ndarray,
                                                 audio_features: any,
                                                 class_names: List[str],
                                                 confidence_threshold: float = 0.7,
                                                 title: Optional[str] = None,
                                                 save_path: Optional[str] = None,
                                                 figsize: Tuple[int, int] = (15, 6)) -> plt.Figure:
        """
        Plot audio waveform with only HIGH CONFIDENCE predictions overlaid.
        
        Args:
            audio_path: Path to audio file
            raw_predictions: Raw prediction array from model
            confidence_scores: Confidence scores for each prediction
            audio_features: Audio features object with meter grid
            class_names: List of class names ['O', 'A', 'B', 'C']
            confidence_threshold: Minimum confidence to show prediction
            title: Optional plot title
            save_path: Optional path to save the plot
            figsize: Figure size (width, height)
            
        Returns:
            matplotlib Figure object
        """
        # Load audio for visualization
        y, sr = librosa.load(audio_path, sr=None)
        y_harm, y_perc = librosa.effects.hpss(y)
        
        plt.close('all')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), dpi=96)
        
        # Top plot: Waveform with high confidence predictions only
        librosa.display.waveshow(y_harm, sr=sr, alpha=0.8, ax=ax1, 
                                color='deepskyblue', label='Harmonic')
        librosa.display.waveshow(y_perc, sr=sr, alpha=0.7, ax=ax1, 
                                color='plum', label='Percussive')
        
        # Calculate meter times
        meter_grid_times = librosa.frames_to_time(
            audio_features.meter_grid, sr=audio_features.sr, hop_length=audio_features.hop_length)
        
        # Track which classes we've added to legend
        legend_added = set()
        
        # Filter predictions by confidence and create pattern
        high_conf_pattern = []
        confident_count = 0
        
        # Highlight only high confidence sections
        for i, prediction in enumerate(raw_predictions):
            if i < len(meter_grid_times) - 1:
                start_time = meter_grid_times[i]
                end_time = meter_grid_times[i + 1]
                section = class_names[prediction]
                confidence = confidence_scores[i]
                
                if confidence >= confidence_threshold:
                    confident_count += 1
                    high_conf_pattern.append(section)
                    
                    color = self.class_colors.get(section, 'black')
                    
                    # Add to legend only once per class
                    label = f"{section} - {self._get_section_description(section)}" if section not in legend_added else None
                    if label:
                        legend_added.add(section)
                    
                    # Use full opacity for high confidence
                    ax1.axvspan(start_time, end_time, color=color, alpha=0.6, label=label)
                else:
                    high_conf_pattern.append('?')  # Uncertain
        
        # Configure top plot
        duration = len(y) / sr
        ax1.set_xlim([0, duration])
        ax1.set_ylabel('Amplitude')
        confidence_pct = (confident_count / len(raw_predictions)) * 100
        ax1.set_title(f'HIGH CONFIDENCE Predictions (≥{confidence_threshold}): {Path(audio_path).stem}\\n{confident_count}/{len(raw_predictions)} segments ({confidence_pct:.1f}%) above threshold')
        ax1.legend(loc='upper right')
        
        # Bottom plot: All confidence scores with threshold line
        segment_times = [(meter_grid_times[i] + meter_grid_times[i+1])/2 
                        for i in range(len(raw_predictions))]
        
        # Color code by whether above threshold
        colors = ['green' if conf >= confidence_threshold else 'red' for conf in confidence_scores]
        ax2.scatter(segment_times, confidence_scores, c=colors, alpha=0.7, s=20)
        ax2.plot(segment_times, confidence_scores, color='gray', alpha=0.5, linewidth=1)
        
        ax2.set_xlim([0, duration])
        ax2.set_ylim([0, 1])
        ax2.set_ylabel('Confidence')
        ax2.set_xlabel('Time (mm:ss)')
        ax2.set_title(f'Confidence Scores (Green = ≥{confidence_threshold}, Red = <{confidence_threshold})')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=confidence_threshold, color='orange', linestyle='--', linewidth=2, label=f'{confidence_threshold} threshold')
        ax2.legend()
        
        # Set time-based x-axis labels for both plots
        xticks = np.arange(0, duration, 30)  # Every 30 seconds
        xlabels = [f"{int(tick // 60)}:{int(tick % 60):02d}" for tick in xticks]
        
        ax1.set_xticks(xticks)
        ax1.set_xticklabels([])  # No labels on top plot
        
        ax2.set_xticks(xticks)
        ax2.set_xticklabels(xlabels)
        
        plt.tight_layout()
        
        # Save if path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ High confidence predictions saved to {save_path}")
        
        # Create simplified pattern string
        high_conf_pattern_str = ''.join(high_conf_pattern)
        print(f"🎯 High confidence pattern: {high_conf_pattern_str}")
        print(f"   {confident_count}/{len(raw_predictions)} segments above {confidence_threshold} confidence")
        
        return fig

    def plot_arrangement_waveform(self,
                                audio_path: str,
                                arrangement_blocks: List[Dict],
                                title: Optional[str] = None,
                                save_path: Optional[str] = None,
                                figsize: Tuple[int, int] = (15, 6)) -> plt.Figure:
        """
        Plot audio waveform with arrangement sections highlighted.

        Args:
            audio_path: Path to audio file
            arrangement_blocks: List of arrangement blocks with sections and timestamps
            title: Optional plot title
            save_path: Optional path to save the plot
            figsize: Figure size (width, height)

        Returns:
            matplotlib Figure object
        """
        # Load audio for visualization
        y, sr = librosa.load(audio_path, sr=None)
        y_harm, y_perc = librosa.effects.hpss(y)

        plt.close('all')
        fig, ax = plt.subplots(figsize=figsize, dpi=96)
        ax.clear()

        # Display waveform components
        librosa.display.waveshow(y_harm, sr=sr, alpha=0.8, ax=ax,
                                color='deepskyblue', label='Harmonic')
        librosa.display.waveshow(y_perc, sr=sr, alpha=0.7, ax=ax,
                                color='plum', label='Percussive')

        # Add vertical lines at block boundaries
        for block in arrangement_blocks:
            start_time = block['start_time']
            ax.axvline(x=start_time, color='grey', linestyle='--',
                      linewidth=1, alpha=0.6)

        # Track which classes we've added to legend
        legend_added = set()

        # Highlight sections by arrangement type
        for block in arrangement_blocks:
            start_time = block['start_time']
            end_time = block['end_time']
            section = block['arrangement_section']

            color = self.class_colors.get(section, 'black')

            # Add to legend only once per class
            label = f"{section} - {self._get_section_description(section)}" if section not in legend_added else None
            if label:
                legend_added.add(section)

            ax.axvspan(start_time, end_time, color=color, alpha=0.4, label=label)

        # Configure plot appearance
        duration = len(y) / sr
        ax.set_xlim([0, duration])
        ax.set_ylabel('Amplitude')

        # Set plot title
        if title:
            ax.set_title(title)
        else:
            audio_file_name = Path(audio_path).stem
            ax.set_title(f'Arrangement Classification: {audio_file_name}')

        # Add legend
        ax.legend(loc='upper right')

        # Set time-based x-axis labels
        xticks = np.arange(0, duration, 30)  # Every 30 seconds
        xlabels = [f"{int(tick // 60)}:{int(tick % 60):02d}" for tick in xticks]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels)
        ax.set_xlabel('Time (mm:ss)')

        plt.tight_layout()

        # Save if path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Visualization saved to {save_path}")

        return fig

    def plot_arrangement_timeline(self,
                                arrangement_blocks: List[Dict],
                                total_duration: float,
                                title: Optional[str] = None,
                                save_path: Optional[str] = None,
                                figsize: Tuple[int, int] = (15, 4)) -> plt.Figure:
        """
        Plot arrangement sections as a colored timeline/bar chart.

        Args:
            arrangement_blocks: List of arrangement blocks
            total_duration: Total track duration in seconds
            title: Optional plot title
            save_path: Optional path to save the plot
            figsize: Figure size (width, height)

        Returns:
            matplotlib Figure object
        """
        plt.close('all')
        fig, ax = plt.subplots(figsize=figsize, dpi=96)

        # Plot bars for each block
        for block in arrangement_blocks:
            start_time = block['start_time']
            end_time = block['end_time']
            section = block['arrangement_section']
            duration = end_time - start_time

            color = self.class_colors.get(section, 'black')
            y_pos = self.class_y_positions.get(section, 0)

            ax.barh(y_pos, duration, left=start_time, height=0.8,
                   color=color, alpha=0.8, edgecolor='white', linewidth=0.5)

            # Add section label in the middle of the bar if it's wide enough
            if duration > 10:  # Only show text for blocks longer than 10 seconds
                ax.text(start_time + duration/2, y_pos, section,
                       ha='center', va='center', fontweight='bold', fontsize=10)

        # Configure plot appearance
        ax.set_xlim([0, total_duration])
        ax.set_ylim([-0.5, 3.5])
        ax.set_yticks(list(self.class_y_positions.values()))

        # Create y-axis labels
        y_labels = []
        for section, y_pos in sorted(self.class_y_positions.items(), key=lambda x: x[1]):
            description = self._get_section_description(section)
            y_labels.append(f"{section}\n{description}")

        ax.set_yticklabels(y_labels)

        # Set time-based x-axis labels
        xticks = np.arange(0, total_duration, 30)  # Every 30 seconds
        xlabels = [f"{int(tick // 60)}:{int(tick % 60):02d}" for tick in xticks]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels)
        ax.set_xlabel('Time (mm:ss)')

        if title:
            ax.set_title(title)
        else:
            ax.set_title('Arrangement Structure Timeline')

        ax.grid(True, axis='x', alpha=0.3)
        plt.tight_layout()

        # Save if path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Timeline saved to {save_path}")

        return fig

    def plot_arrangement_comparison(self,
                                  audio_path: str,
                                  arrangement_blocks: List[Dict],
                                  detailed_pattern: str,
                                  simplified_pattern: str,
                                  save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a comprehensive visualization showing both waveform and timeline
        with detailed vs simplified patterns.

        Args:
            audio_path: Path to audio file
            arrangement_blocks: List of arrangement blocks
            detailed_pattern: Detailed pattern string (with *)
            simplified_pattern: Simplified pattern string
            save_path: Optional path to save the plot

        Returns:
            matplotlib Figure object
        """
        # Load audio for duration
        y, sr = librosa.load(audio_path, sr=None)
        total_duration = len(y) / sr

        plt.close('all')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), dpi=96)

        # Top plot: Waveform with sections
        y_harm, y_perc = librosa.effects.hpss(y)
        librosa.display.waveshow(y_harm, sr=sr, alpha=0.8, ax=ax1,
                                color='deepskyblue', label='Harmonic')
        librosa.display.waveshow(y_perc, sr=sr, alpha=0.7, ax=ax1,
                                color='plum', label='Percussive')

        # Add arrangement sections to waveform
        legend_added = set()
        for block in arrangement_blocks:
            start_time = block['start_time']
            end_time = block['end_time']
            section = block['arrangement_section']

            color = self.class_colors.get(section, 'black')
            label = f"{section} - {self._get_section_description(section)}" if section not in legend_added else None
            if label:
                legend_added.add(section)

            ax1.axvspan(start_time, end_time, color=color, alpha=0.4, label=label)

        ax1.set_xlim([0, total_duration])
        ax1.set_ylabel('Amplitude')
        ax1.set_title(f'Audio Waveform with Arrangement Sections\\n{Path(audio_path).stem}')
        ax1.legend(loc='upper right')

        # Bottom plot: Timeline
        for block in arrangement_blocks:
            start_time = block['start_time']
            end_time = block['end_time']
            section = block['arrangement_section']
            duration = end_time - start_time

            color = self.class_colors.get(section, 'black')
            y_pos = self.class_y_positions.get(section, 0)

            ax2.barh(y_pos, duration, left=start_time, height=0.8,
                   color=color, alpha=0.8, edgecolor='white', linewidth=0.5)

            # Add section label for longer blocks
            if duration > 8:
                ax2.text(start_time + duration/2, y_pos, section, 
                        ha='center', va='center', fontweight='bold', fontsize=9)

        ax2.set_xlim([0, total_duration])
        ax2.set_ylim([-0.5, 3.5])
        ax2.set_yticks(list(self.class_y_positions.values()))

        y_labels = []
        for section, y_pos in sorted(self.class_y_positions.items(), key=lambda x: x[1]):
            description = self._get_section_description(section)
            y_labels.append(f"{section}\\n{description}")
        ax2.set_yticklabels(y_labels)

        # Set time labels for both plots
        xticks = np.arange(0, total_duration, 30)
        xlabels = [f"{int(tick // 60)}:{int(tick % 60):02d}" for tick in xticks]

        ax1.set_xticks(xticks)
        ax1.set_xticklabels([])  # No labels on top plot

        ax2.set_xticks(xticks)
        ax2.set_xticklabels(xlabels)
        ax2.set_xlabel('Time (mm:ss)')

        # Add pattern information
        pattern_text = f"Detailed: {detailed_pattern}\\nSimplified: {simplified_pattern}"
        ax2.set_title(f'Arrangement Timeline\\n{pattern_text}')
        ax2.grid(True, axis='x', alpha=0.3)

        plt.tight_layout()

        # Save if path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Comparison visualization saved to {save_path}")

        return fig

    def _get_section_description(self, section: str) -> str:
        """Get human-readable description for section code."""
        descriptions = {
            'O': 'Intro/Outro/Other',
            'A': 'Medium Energy',
            'B': 'High Energy',
            'C': 'Breakdown'
        }
        return descriptions.get(section, 'Unknown')

    def show_plot(self, fig: plt.Figure):
        """Display the plot (useful for interactive environments)."""
        plt.show()

    def close_all_plots(self):
        """Close all matplotlib plots to free memory."""
        plt.close('all')


# Example usage function
def visualize_arrangement_analysis(audio_path: str,
                                 arrangement_blocks: List[Dict],
                                 detailed_pattern: str,
                                 simplified_pattern: str,
                                 output_dir: Optional[str] = None) -> None:
    """
    Convenience function to create all visualizations for an arrangement analysis.

    Args:
        audio_path: Path to audio file
        arrangement_blocks: List of arrangement blocks from classifier
        detailed_pattern: Detailed pattern with * markers
        simplified_pattern: Simplified pattern
        output_dir: Optional directory to save visualizations
    """
    visualizer = SongVisualizerService()
    audio_name = Path(audio_path).stem

    try:
        # Create waveform visualization
        waveform_fig = visualizer.plot_arrangement_waveform(
            audio_path=audio_path,
            arrangement_blocks=arrangement_blocks,
            save_path=os.path.join(output_dir, f"{audio_name}_waveform.png") if output_dir else None
        )

        # Create timeline visualization
        total_duration = max(block['end_time'] for block in arrangement_blocks) if arrangement_blocks else 0
        timeline_fig = visualizer.plot_arrangement_timeline(
            arrangement_blocks=arrangement_blocks,
            total_duration=total_duration,
            save_path=os.path.join(output_dir, f"{audio_name}_timeline.png") if output_dir else None
        )

        # Create comparison visualization
        comparison_fig = visualizer.plot_arrangement_comparison(
            audio_path=audio_path,
            arrangement_blocks=arrangement_blocks,
            detailed_pattern=detailed_pattern,
            simplified_pattern=simplified_pattern,
            save_path=os.path.join(output_dir, f"{audio_name}_comparison.png") if output_dir else None
        )

        print(f"✅ Created visualizations for {audio_name}")

        # Show plots (comment out if running in batch mode)
        visualizer.show_plot(comparison_fig)

    except Exception as e:
        print(f"❌ Error creating visualizations: {e}")
    finally:
        visualizer.close_all_plots()
"""
Audio processing functionality for chorus detection.
"""

import numpy as np
from typing import List, Tuple
import librosa
from sklearn.preprocessing import StandardScaler
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from functools import reduce


# Constants
SR = 12000
HOP_LENGTH = 128
MAX_FRAMES = 300
MAX_METERS = 201
N_FEATURES = 15


class AudioFeature:
    """Class for extracting and processing audio features."""

    def __init__(self, audio_path, sr=SR, hop_length=HOP_LENGTH):
        self.audio_path = audio_path
        self.sr = sr
        self.hop_length = hop_length
        self.y = None
        self.y_harm = self.y_perc = None
        self.beats = None
        self.chromagram = self.chroma_acts = None
        self.combined_features = None
        self.key = self.mode = None
        self.mel_acts = self.melspectrogram = None
        self.meter_grid = None
        self.mfccs = self.mfcc_acts = None
        self.n_frames = None
        self.onset_env = None
        self.rms = None
        self.spectrogram = None
        self.tempo = None
        self.tempogram = self.tempogram_acts = None
        self.time_signature = 4

    def detect_key(self, chroma_vals: np.ndarray) -> Tuple[str, str]:
        """Detect the key and mode (major or minor) of the audio segment."""
        note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        major_profile = np.array(
            [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
        )
        minor_profile = np.array(
            [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
        )

        # Normalize profiles
        major_profile /= np.linalg.norm(major_profile)
        minor_profile /= np.linalg.norm(minor_profile)

        # Calculate correlations for all possible keys
        major_correlations = [
            np.corrcoef(chroma_vals, np.roll(major_profile, i))[0, 1] for i in range(12)
        ]
        minor_correlations = [
            np.corrcoef(chroma_vals, np.roll(minor_profile, i))[0, 1] for i in range(12)
        ]

        # Find best match
        max_major_idx = np.argmax(major_correlations)
        max_minor_idx = np.argmax(minor_correlations)

        self.mode = (
            "major"
            if major_correlations[max_major_idx] > minor_correlations[max_minor_idx]
            else "minor"
        )
        self.key = note_names[max_major_idx if self.mode == "major" else max_minor_idx]
        return self.key, self.mode

    def calculate_ki_chroma(
        self, waveform: np.ndarray, sr: int, hop_length: int
    ) -> np.ndarray:
        """Calculate a normalized, key-invariant chromagram."""
        chromagram = librosa.feature.chroma_cqt(
            y=waveform, sr=sr, hop_length=hop_length, bins_per_octave=24
        )
        chromagram = (chromagram - chromagram.min()) / (
            chromagram.max() - chromagram.min()
        )

        chroma_vals = np.sum(chromagram, axis=1)
        key, mode = self.detect_key(chroma_vals)

        key_idx = [
            "C",
            "C#",
            "D",
            "D#",
            "E",
            "F",
            "F#",
            "G",
            "G#",
            "A",
            "A#",
            "B",
        ].index(key)
        shift_amount = -key_idx if mode == "major" else -(key_idx + 3) % 12

        return librosa.util.normalize(np.roll(chromagram, shift_amount, axis=0), axis=1)

    def extract_features_from_audio(self):
        """
        Extract features from audio file using the same process as training.
        """
        print(f"Processing: {self.audio_path}")

        # Load audio and separate harmonic/percussive components
        self.y, self.sr = librosa.load(self.audio_path, sr=self.sr)
        print(f"Loaded audio: {len(self.y)/self.sr:.1f} seconds")

        self.y_harm, self.y_perc = librosa.effects.hpss(self.y)

        # Extract spectrogram and RMS
        self.spectrogram, _ = librosa.magphase(
            librosa.stft(self.y, hop_length=self.hop_length)
        )
        self.rms = librosa.feature.rms(
            S=self.spectrogram, hop_length=self.hop_length
        ).astype(np.float32)

        # Extract mel spectrogram and its components
        self.melspectrogram = librosa.feature.melspectrogram(
            y=self.y, sr=self.sr, n_mels=128, hop_length=self.hop_length
        ).astype(np.float32)
        self.mel_acts = librosa.decompose.decompose(
            self.melspectrogram, n_components=3, sort=True
        )[1].astype(np.float32)

        # Extract chromagram and its components
        self.chromagram = self.calculate_ki_chroma(
            self.y_harm, self.sr, self.hop_length
        ).astype(np.float32)
        self.chroma_acts = librosa.decompose.decompose(
            self.chromagram, n_components=4, sort=True
        )[1].astype(np.float32)

        # Extract onset envelope and tempogram
        self.onset_env = librosa.onset.onset_strength(
            y=self.y_perc, sr=self.sr, hop_length=self.hop_length
        )
        self.tempogram = np.clip(
            librosa.feature.tempogram(
                onset_envelope=self.onset_env, sr=self.sr, hop_length=self.hop_length
            ),
            0,
            None,
        )
        self.tempogram_acts = librosa.decompose.decompose(
            self.tempogram, n_components=3, sort=True
        )[1]

        # Extract MFCCs and components
        self.mfccs = librosa.feature.mfcc(
            y=self.y, sr=self.sr, n_mfcc=20, hop_length=self.hop_length
        )
        self.mfccs += abs(np.min(self.mfccs))
        self.mfcc_acts = librosa.decompose.decompose(
            self.mfccs, n_components=4, sort=True
        )[1].astype(np.float32)

        # Combine features with weighted normalization
        features = [
            self.rms,
            self.mel_acts,
            self.chroma_acts,
            self.tempogram_acts,
            self.mfcc_acts,
        ]
        feature_names = [
            "rms",
            "mel_acts",
            "chroma_acts",
            "tempogram_acts",
            "mfcc_acts",
        ]

        # Calculate weights for each feature type
        dims = {
            name: feature.shape[0] for feature, name in zip(features, feature_names)
        }
        total_inv_dim = sum(1 / dim for dim in dims.values())
        weights = {name: 1 / (dims[name] * total_inv_dim) for name in feature_names}

        # Standardize and weight features
        std_weighted_features = [
            StandardScaler().fit_transform(feature.T).T * weights[name]
            for feature, name in zip(features, feature_names)
        ]

        self.combined_features = np.concatenate(std_weighted_features, axis=0).T.astype(
            np.float32
        )
        self.n_frames = len(self.combined_features)

    def create_meter_grid(self):
        """Create a grid based on the meter of the song, using tempo and beats."""
        self.tempo, self.beats = librosa.beat.beat_track(
            onset_envelope=self.onset_env, sr=self.sr, hop_length=self.hop_length
        )

        # Adjust tempo to reasonable range
        if self.tempo < 75:
            self.tempo *= 2
        elif self.tempo > 160:
            self.tempo /= 2

        self.meter_grid = self._create_meter_grid()
        return self.meter_grid

    def _create_meter_grid(self, target_segment_seconds=5) -> np.ndarray:
        """
        Create a simple grid with fixed 5-second segments to match training data.
        """
        time_duration = librosa.frames_to_time(
            self.n_frames, sr=self.sr, hop_length=self.hop_length
        )
        
        # Create simple 5-second segments starting from 0
        segment_times = np.arange(0, time_duration + target_segment_seconds, target_segment_seconds)
        
        # Ensure we don't go beyond the actual audio duration
        segment_times = segment_times[segment_times <= time_duration]
        
        # Always end exactly at the audio duration
        if segment_times[-1] < time_duration:
            segment_times = np.append(segment_times, time_duration)

        # Convert to frames
        meter_grid_frames = librosa.time_to_frames(
            segment_times, sr=self.sr, hop_length=self.hop_length
        )

        # Ensure final frame is included
        if meter_grid_frames[-1] != self.n_frames:
            meter_grid_frames[-1] = self.n_frames

        print(f"Created meter grid with {len(meter_grid_frames)-1} segments")
        print(f"Audio duration: {time_duration:.1f}s")
        print(f"Expected segments (~5s each): {time_duration/5:.1f}")
        print(f"Actual segments created: {len(meter_grid_frames)-1}")
        print(f"Segment duration target: {target_segment_seconds:.1f}s")
        
        segment_lengths = np.diff(
            librosa.frames_to_time(
                meter_grid_frames, sr=self.sr, hop_length=self.hop_length
            )
        )
        print(
            f"Segment lengths: min={segment_lengths.min():.1f}s, max={segment_lengths.max():.1f}s, mean={segment_lengths.mean():.1f}s"
        )
        
        # Debug the actual segment times
        segment_times = librosa.frames_to_time(meter_grid_frames, sr=self.sr, hop_length=self.hop_length)
        print(f"First 5 segment boundaries: {segment_times[:5]}")
        print(f"Last 5 segment boundaries: {segment_times[-5:]}")

        return meter_grid_frames


class AudioMeter:
    def __init__(self):
        pass

    def segment_data_meters(
        self, data: np.ndarray, meter_grid: List[int]
    ) -> List[np.ndarray]:
        """Segment input data into chunks based on a meter grid."""
        return [
            data[meter_grid[i] : meter_grid[i + 1]] for i in range(len(meter_grid) - 1)
        ]

    def positional_encoding(self, position: int, d_model: int) -> np.ndarray:
        """Add positional encoding to input data."""
        pe = np.zeros(d_model)
        for i in range(0, d_model, 2):
            pe[i] = np.sin(position / (10000 ** (i / d_model)))
            if i + 1 < d_model:
                pe[i + 1] = np.cos(position / (10000 ** (i / d_model)))
        return pe

    def apply_hierarchical_positional_encoding(
        self, segments: List[np.ndarray]
    ) -> List[np.ndarray]:
        """Apply positional encoding to a list of segments."""
        encoded_segments = []
        for meter_idx, meter_segment in enumerate(segments):
            meter_encoded = np.zeros_like(meter_segment)
            for frame_idx, frame in enumerate(meter_segment):
                frame_pos_encoding = (
                    self.positional_encoding(frame_idx, frame.shape[0]) * 0.1
                )
                meter_pos_encoding = (
                    self.positional_encoding(meter_idx, frame.shape[0]) * 0.2
                )
                meter_encoded[frame_idx] = (
                    frame + frame_pos_encoding + meter_pos_encoding
                )
            encoded_segments.append(meter_encoded)
        return encoded_segments


class AudioProcess:

    def __init__(self):
        pass

    def strip_silence(self, audio_path):
        """Removes silent parts from an audio file."""
        sound = AudioSegment.from_file(audio_path)
        nonsilent_ranges = detect_nonsilent(
            sound, min_silence_len=500, silence_thresh=-50
        )
        stripped = reduce(
            lambda acc, val: acc + sound[val[0] : val[1]],
            nonsilent_ranges,
            AudioSegment.empty(),
        )
        stripped.export(audio_path, format="mp3")

    def pad_song(
        self,
        encoded_segments: List[np.ndarray],
        max_frames: int = MAX_FRAMES,
        max_meters: int = MAX_METERS,
        n_features: int = N_FEATURES,
    ) -> np.ndarray:
        """
        Pad a list of encoded segments to create a uniform 3D array.

        Parameters:
        - encoded_segments (list): List of encoded data segments
        - max_frames (int): Maximum number of frames per segment
        - max_meters (int): Maximum number of meters
        - n_features (int): Number of features per frame

        Returns:
        - np.ndarray: Padded 3D array of shape (max_meters, max_frames, n_features)
        """
        padded_song = np.zeros((max_meters, max_frames, n_features))

        for i, segment in enumerate(encoded_segments):
            if i >= max_meters:
                break  # Only consider up to max_meters segments

            segment_frames = segment.shape[0]
            if segment_frames <= max_frames:
                # If segment fits, copy it directly
                padded_song[i, :segment_frames, :] = segment
            else:
                # If segment is too long, sample frames evenly
                indices = np.linspace(0, segment_frames - 1, max_frames, dtype=int)
                padded_song[i, :, :] = segment[indices, :]

        return padded_song

    def process_audio(
        self, audio_path, trim_silence=True, sr=SR, hop_length=HOP_LENGTH
    ):
        """Process an audio file for chorus detection."""
        try:
            # Optionally strip silence
            if trim_silence:
                self.strip_silence(audio_path)

            # Extract audio features
            audio_features = AudioFeature(audio_path, sr=sr, hop_length=hop_length)
            audio_features.extract_features_from_audio()
            meter_grid = audio_features.create_meter_grid()

            # Segment and pad the data
            audio_meter = AudioMeter()
            feature_segments = audio_meter.segment_data_meters(
                audio_features.combined_features, meter_grid
            )
            encoded_segments = audio_meter.apply_hierarchical_positional_encoding(
                feature_segments
            )
            padded_song = self.pad_song(encoded_segments)

            # Add batch dimension for model
            padded_song = np.expand_dims(padded_song, axis=0)
            return padded_song, audio_features
        except Exception as e:
            print(f"Error processing audio: {e}")
            return None, None


# padded_song, audio_features = process_audio(full_path_2)

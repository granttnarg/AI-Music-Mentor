# Pipeline idea: Upload → Feature Extraction → Embeddings → Data Object → Comparison/RAG

import librosa
import os
import numpy as np


class AudioFeatureService:

    def __init__(self, sr=22050, hop_length=128):
        self.sr = sr
        self.hop_length = hop_length
        self.tempo = None
        self.audio_path = None

        self.y = None  # amplitude
        self.y_harm = None  # harmonic
        self.y_perc = None  # rhythmic

    def load_audio_file(self, audio_path):
        """
        Load file from audio path and return amplitude array.
        """
        self.audio_path = audio_path

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        self.audio_file = audio_path
        self.y, _ = librosa.load(audio_path, sr=self.sr)
        print("file successfully loaded!")

        return self

    def extract_global_features(self, max_duration=600):
        """
        Extract global (whole-track) audio features from a song
        No segmentation - just overall track characteristics.
        """
        y, y_perc, y_harm, sr, hop_length, duration = self._prepare_audio(max_duration)

        global_features = {
            "metadata": {"duration": duration, "sample_rate": sr},
            "rhythm": self._extract_rhythm_features(y_perc, sr, hop_length, duration),
            "harmony": self._extract_harmony_features(y_harm, sr, hop_length, duration),
            "energy": self._extract_energy_features(y, hop_length, duration),
            "spectral": self._extract_spectral_features(y, sr, hop_length),
            "frequency": self._extract_frequency_features(y, sr, hop_length),
        }

        print(f"## Global features extracted for {self.audio_path}!")
        return global_features

    def create_embedding_vector(self, feature_data, exclude_categories=None):
        """
        Convert feature_data to raw vector with basic normalization.
        Filtered sections will be replaced with default values to keep vector size.

        Args:
            feature_data: Output from extract_global_features() or filter_feature_set()

        Returns:
            np.array of 19 normalized features (0-1 range mostly)

        Note: Missing categories get neutral defaults to maintain consistent
              vector dimensions for similarity calculations.
        """

        if exclude_categories:
            feature_data = self.filter_feature_set(feature_data, exclude_categories)
        else:
            feature_data = feature_data

        # Extract raw features (removed arbitrary scaling)
        vector = np.array(
            [
                # Rhythm - with safe defaults
                feature_data.get("rhythm", {}).get("tempo", 120) / 200.0,
                feature_data.get("rhythm", {}).get("onset_density", 0) / 15.0,
                feature_data.get("rhythm", {}).get("syncopation_level", 0),
                feature_data.get("rhythm", {}).get("rhythmic_variance", 0) / 0.1,
                # Harmony
                feature_data.get("harmony", {}).get("chroma_variance", 0) / 0.1,
                feature_data.get("harmony", {}).get("key_strength", 1) / 3.0,
                feature_data.get("harmony", {}).get("harmonic_change_rate", 0) / 0.005,
                feature_data.get("harmony", {}).get("tonal_stability", 0.5),
                # Energy
                feature_data.get("energy", {}).get("energy_range", 0),
                feature_data.get("energy", {}).get("avg_energy", 0),
                abs(feature_data.get("energy", {}).get("energy_trend", 0)) / 0.001,
                feature_data.get("energy", {}).get("peak_density", 0) / 25.0,
                # Spectral
                feature_data.get("spectral", {}).get("avg_brightness", 1000) / 8000.0,
                feature_data.get("spectral", {}).get("brightness_variance", 0)
                / 2000000.0,
                # Frequency
                feature_data.get("frequency", {}).get("low_proportion", 0.33),
                feature_data.get("frequency", {}).get("mid_proportion", 0.33),
                feature_data.get("frequency", {}).get("high_proportion", 0.33),
                feature_data.get("frequency", {}).get("mid_low_ratio", 1.0),
                feature_data.get("frequency", {}).get("high_mid_ratio", 1.0),
            ]
        )

        return vector

    def filter_feature_set(self, feature_data, exclude_categories=["spectral"]):
        """
        Filter feature data by excluding specified categories.

        Args:
            feature_data: Output from extract_global_features()
            exclude_categories: List of categories to remove
                              (from: 'rhythm', 'harmony', 'energy', 'spectral', 'frequency')

        Returns:
            Filtered feature data with same structure but excluded categories removed
        """
        if not exclude_categories:
            return feature_data

        # Create a copy to avoid modifying original data
        filtered_data = feature_data.copy()

        # Remove excluded categories
        for category in exclude_categories:
            if category in filtered_data:
                del filtered_data[category]

        return filtered_data

    def build_feature_data_object(
        self, feature_data, categories=["eq", "energy", "rhythm"]
    ):
        """
        Reorganize global features into feedback-oriented categories for RAG/DB storage.

        Args:
            feature_data: Output from extract_global_features()
            categories: List of feedback categories to include
                      (available: 'eq', 'energy', 'rhythm', 'arrangement')

        Returns:
            Dict organized by feedback categories with relevant features
        """
        feedback_object = {}

        # Add metadata
        if "metadata" in feature_data:
            feedback_object["metadata"] = feature_data["metadata"].copy()

        # EQ category - frequency balance and spectral characteristics
        if "eq" in categories:
            feedback_object["eq"] = {
                "brightness": feature_data.get("spectral", {}).get("avg_brightness", 0),
                "brightness_variance": feature_data.get("spectral", {}).get(
                    "brightness_variance", 0
                ),
                "low_proportion": feature_data.get("frequency", {}).get(
                    "low_proportion", 0
                ),
                "mid_proportion": feature_data.get("frequency", {}).get(
                    "mid_proportion", 0
                ),
                "high_proportion": feature_data.get("frequency", {}).get(
                    "high_proportion", 0
                ),
                "rolloff_frequency": feature_data.get("spectral", {}).get(
                    "avg_rolloff", 0
                ),
                "spectral_bandwidth": feature_data.get("spectral", {}).get(
                    "avg_bandwidth", 0
                ),
                "mid_low_ratio": feature_data.get("frequency", {}).get(
                    "mid_low_ratio", 0
                ),
                "high_mid_ratio": feature_data.get("frequency", {}).get(
                    "high_mid_ratio", 0
                ),
            }

        # Energy category - dynamics and loudness
        if "energy" in categories:
            feedback_object["energy"] = {
                "dynamic_range": feature_data.get("energy", {}).get("energy_range", 0),
                "average_energy": feature_data.get("energy", {}).get("avg_energy", 0),
                "energy_trend": feature_data.get("energy", {}).get("energy_trend", 0),
                "peak_density": feature_data.get("energy", {}).get("peak_density", 0),
                "beat_strength": feature_data.get("rhythm", {}).get("beat_strength", 0),
            }

        # Rhythm category - timing and groove
        if "rhythm" in categories:
            feedback_object["rhythm"] = {
                "tempo": feature_data.get("rhythm", {}).get("tempo", 0),
                "onset_density": feature_data.get("rhythm", {}).get("onset_density", 0),
                "syncopation_level": feature_data.get("rhythm", {}).get(
                    "syncopation_level", 0
                ),
                "rhythmic_variance": feature_data.get("rhythm", {}).get(
                    "rhythmic_variance", 0
                ),
                "beat_strength": feature_data.get("rhythm", {}).get("beat_strength", 0),
            }

        return feedback_object

    # ======================================================================================= #
    # PRIVATE METHODS #

    def _prepare_audio(self, max_duration):
        """Prepare audio for feature extraction"""
        y = self.y
        sr = self.sr
        hop_length = self.hop_length

        if max_duration:
            max_samples = int(max_duration * sr)
            y = y[:max_samples]

        # Extract harmonic and rhythmic material
        duration = float(librosa.get_duration(y=y, sr=sr))
        y_harm, y_perc = librosa.effects.hpss(y)

        # Store separated audio for use in feature extraction
        self.y_harm = y_harm
        self.y_perc = y_perc

        print("Audio Prepped")
        return y, y_perc, y_harm, sr, hop_length, duration

    def _extract_rhythm_features(self, y_perc, sr, hop_length, duration):
        """Extract rhythm-related features"""

        # Tempo and beat consistency
        tempo, _beats = librosa.beat.beat_track(y=y_perc, sr=sr, hop_length=hop_length)
        tempo = float(tempo)

        # Save BPM for other methods
        self.tempo = tempo

        # Onset feature_data
        onset_env = librosa.onset.onset_strength(y=y_perc, sr=sr, hop_length=hop_length)
        onsets = librosa.onset.onset_detect(
            onset_envelope=onset_env, sr=sr, hop_length=hop_length
        )
        onset_times = librosa.frames_to_time(onsets, sr=sr, hop_length=hop_length)

        # Rhythmic complexity metrics
        if len(onset_times) > 1:
            intervals = np.diff(onset_times)
            beat_duration = 60.0 / tempo
            beat_relative = intervals / beat_duration
            syncopation = np.mean(np.abs(beat_relative - np.round(beat_relative)))
            rhythmic_variance = np.var(intervals)
            onset_density = len(onset_times) / duration
        else:
            syncopation = 0
            rhythmic_variance = 0
            onset_density = 0

        print("Extract Rhythm Features")
        return {
            "tempo": tempo,
            "onset_density": onset_density,
            "syncopation_level": syncopation,
            "rhythmic_variance": rhythmic_variance,
            "beat_strength": np.mean(onset_env),
        }

    def _extract_harmony_features(self, y_harm, sr, hop_length, duration):
        """Extract harmony-related features"""

        # Chroma feature_data
        chroma = librosa.feature.chroma_cqt(y=y_harm, sr=sr, hop_length=hop_length)
        chroma_mean = np.mean(chroma, axis=1)

        # Key strength and harmonic complexity
        key_strength = np.max(chroma_mean) / (np.mean(chroma_mean) + 1e-8)
        chroma_variance = np.var(chroma, axis=1).mean()

        # Harmonic change rate
        chroma_diff = np.diff(chroma, axis=1)
        harmonic_change_rate = np.mean(np.sum(np.abs(chroma_diff), axis=0)) / duration

        print("Extract Harmonic Features")
        return {
            "chroma_variance": chroma_variance,
            "key_strength": key_strength,
            "harmonic_change_rate": harmonic_change_rate,
            "tonal_stability": 1.0 - np.std(chroma_mean),
        }

    def _extract_energy_features(self, y, hop_length, duration):
        """Extract energy-related features"""
        # Energy dynamics
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        energy_range = np.max(rms) - np.min(rms)
        avg_energy = np.mean(rms)

        # Energy curve shape
        energy_trend = np.polyfit(range(len(rms)), rms, 1)[0]

        # Peak feature_data
        from scipy.signal import find_peaks

        peaks, _ = find_peaks(rms, height=np.mean(rms))
        peak_density = len(peaks) / duration

        print("Extract Energy Features")
        return {
            "energy_range": energy_range,
            "avg_energy": avg_energy,
            "energy_trend": energy_trend,
            "peak_density": peak_density,
        }

    def _extract_spectral_features(self, y, sr, hop_length):
        """Extract spectral characteristics"""
        # Overall spectral characteristics
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[
            0
        ]
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length)[0]
        bandwidth = librosa.feature.spectral_bandwidth(
            y=y, sr=sr, hop_length=hop_length
        )[0]

        print("Extract Spectral Features")

        return {
            "avg_brightness": np.mean(centroid),
            "brightness_variance": np.var(centroid),
            "avg_rolloff": np.mean(rolloff),
            "avg_bandwidth": np.mean(bandwidth),
        }

    def _extract_frequency_features(self, y, sr, hop_length):
        """Extract frequency band feature_data"""
        # Frequency content feature_data
        S = np.abs(librosa.stft(y, hop_length=hop_length))
        freqs = librosa.fft_frequencies(sr=sr)

        # Define frequency bands
        low_band = (freqs >= 20) & (freqs <= 250)  # Bass/kick
        mid_band = (freqs >= 250) & (freqs <= 2000)  # Vocals/snares
        high_band = (freqs >= 2000) & (freqs <= 8000)  # Cymbals/air

        # Calculate average energy in each band
        low_energy = np.mean(S[low_band])
        mid_energy = np.mean(S[mid_band])
        high_energy = np.mean(S[high_band])

        total_energy = low_energy + mid_energy + high_energy

        print("Extract Frequency Features")

        return {
            "low_proportion": low_energy / total_energy,
            "mid_proportion": mid_energy / total_energy,
            "high_proportion": high_energy / total_energy,
            "mid_low_ratio": mid_energy / (low_energy + 1e-8),
            "high_mid_ratio": high_energy / (mid_energy + 1e-8),
        }

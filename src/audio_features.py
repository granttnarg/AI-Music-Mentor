
# Pipeline idea: Upload → Feature Extraction → Embeddings → Data Object → Comparison/RAG

import librosa
import os
import numpy as np


class AudioFeatureService:

  def __init__(self, sr=22050, hop_length= 128):
    self.sr = sr
    self.hop_length =  hop_length
    self.y = None
    self.tempo = None
    self.audio_path = None

  def load_audio_file(self, audio_path):
    """
      Load file from audio path and return amplitude array.
    """
    self.audio_path = audio_path

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")


    self.audio_file = audio_path
    self.y, _ = librosa.load(audio_path, sr=self.sr)
    print('file successfully loaded!')

    return self


  def extract_global_features(self, max_duration=300):
    """
    Extract global (whole-track) audio features from a song
    No segmentation - just overall track characteristics.
    """
    y, sr, hop_length, duration = self._prepare_audio(max_duration)

    global_features = {
        'metadata': {
            'duration': duration,
            'sample_rate': sr
        },
        'rhythm': self._extract_rhythm_features(y, sr, hop_length, duration),
        'harmony': self._extract_harmony_features(sr, hop_length, duration),
        'energy': self._extract_energy_features(y, hop_length, duration),
        'spectral': self._extract_spectral_features(y, sr, hop_length),
        'frequency': self._extract_frequency_features(y, sr, hop_length)
    }

    print(f'## Global features extracted for {self.audio_path}!')
    return global_features

  def filter_feature_set(self, analysis, exclude_categories=['spectral']):
    pass

  def create_embedding_vector(self, analysis):
    pass

  def build_feature_data_object(self, analysis, categories=['eq', 'energy', 'rhythm']):
    pass


  ## PRIVATE METHODS

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

      print('Audio Prepped')
      return y, sr, hop_length, duration

  def _extract_rhythm_features(self, y, sr, hop_length, duration):
      """Extract rhythm-related features"""
      y_perc = self.y_perc

      # Tempo and beat consistency
      tempo, beats = librosa.beat.beat_track(y=y_perc, sr=sr, hop_length=hop_length)
      tempo = float(tempo)

      # Save BPM for other methods
      self.tempo = tempo

      # Onset analysis
      onset_env = librosa.onset.onset_strength(y=y_perc, sr=sr, hop_length=hop_length)
      onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
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

      print('Extract Rhythm Features')
      return {
          'tempo': tempo,
          'onset_density': onset_density,
          'syncopation_level': syncopation,
          'rhythmic_variance': rhythmic_variance,
          'beat_strength': np.mean(onset_env)
      }

  def _extract_harmony_features(self, sr, hop_length, duration):
      """Extract harmony-related features"""
      y_harm = self.y_harm

      # Chroma analysis
      chroma = librosa.feature.chroma_cqt(y=y_harm, sr=sr, hop_length=hop_length)
      chroma_mean = np.mean(chroma, axis=1)

      # Key strength and harmonic complexity
      key_strength = np.max(chroma_mean) / (np.mean(chroma_mean) + 1e-8)
      chroma_variance = np.var(chroma, axis=1).mean()

      # Harmonic change rate
      chroma_diff = np.diff(chroma, axis=1)
      harmonic_change_rate = np.mean(np.sum(np.abs(chroma_diff), axis=0)) / duration

      print('Extract Harmonic Features')
      return {
          'chroma_variance': chroma_variance,
          'key_strength': key_strength,
          'harmonic_change_rate': harmonic_change_rate,
          'tonal_stability': 1.0 - np.std(chroma_mean)
      }

  def _extract_energy_features(self, y, hop_length, duration):
      """Extract energy-related features"""
      # Energy dynamics
      rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
      energy_range = np.max(rms) - np.min(rms)
      avg_energy = np.mean(rms)

      # Energy curve shape
      energy_trend = np.polyfit(range(len(rms)), rms, 1)[0]

      # Peak analysis
      from scipy.signal import find_peaks
      peaks, _ = find_peaks(rms, height=np.mean(rms))
      peak_density = len(peaks) / duration

      print('Extract Energy Features')
      return {
          'energy_range': energy_range,
          'avg_energy': avg_energy,
          'energy_trend': energy_trend,
          'peak_density': peak_density
      }

  def _extract_spectral_features(self, y, sr, hop_length):
      """Extract spectral characteristics"""
      # Overall spectral characteristics
      centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
      rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length)[0]
      bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length)[0]

      print('Extract Spectral Features')

      return {
          'avg_brightness': np.mean(centroid),
          'brightness_variance': np.var(centroid),
          'avg_rolloff': np.mean(rolloff),
          'avg_bandwidth': np.mean(bandwidth)
      }

  def _extract_frequency_features(self, y, sr, hop_length):
      """Extract frequency band analysis"""
      # Frequency content analysis
      S = np.abs(librosa.stft(y, hop_length=hop_length))
      freqs = librosa.fft_frequencies(sr=sr)

      # Define frequency bands
      low_band = (freqs >= 20) & (freqs <= 250)     # Bass/kick
      mid_band = (freqs >= 250) & (freqs <= 2000)   # Vocals/snares
      high_band = (freqs >= 2000) & (freqs <= 8000) # Cymbals/air

      # Calculate average energy in each band
      low_energy = np.mean(S[low_band])
      mid_energy = np.mean(S[mid_band])
      high_energy = np.mean(S[high_band])

      total_energy = low_energy + mid_energy + high_energy

      print('Extract Frequency Features')

      return {
          'low_proportion': low_energy / total_energy,
          'mid_proportion': mid_energy / total_energy,
          'high_proportion': high_energy / total_energy,
          'mid_low_ratio': mid_energy / (low_energy + 1e-8),
          'high_mid_ratio': high_energy / (mid_energy + 1e-8)
      }
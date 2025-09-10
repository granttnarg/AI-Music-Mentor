import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from unittest.mock import Mock, patch
from src.audio_features import AudioFeatureService


class TestAudioFeatureService:
    
    @pytest.fixture
    def mock_audio_data(self):
        """Create mock audio data for testing"""
        # Generate 30 seconds of mock audio at 22050 Hz
        duration = 30
        sr = 22050
        samples = duration * sr
        return np.random.random(samples).astype(np.float32)
    
    @pytest.fixture
    def service(self):
        """Create AudioFeatureService instance"""
        return AudioFeatureService()
    
    @patch('librosa.load')
    @patch('os.path.exists')
    def test_global_feature_extraction_structure(self, mock_exists, mock_load, service, mock_audio_data):
        """Test that extract_global_features returns correct structure and data types"""
        # Setup mocks
        mock_exists.return_value = True
        mock_load.return_value = (mock_audio_data, 22050)
        
        # Load mock audio file
        service.load_audio_file("fake_path.mp3")
        
        # Extract global features
        features = service.extract_global_features(max_duration=30)
        
        # Test top-level structure
        expected_categories = ['metadata', 'rhythm', 'harmony', 'energy', 'spectral', 'frequency']
        assert isinstance(features, dict)
        for category in expected_categories:
            assert category in features, f"Missing category: {category}"
        
        # Test metadata structure and types
        metadata = features['metadata']
        assert isinstance(metadata['duration'], float)
        assert isinstance(metadata['sample_rate'], int)
        assert metadata['duration'] > 0
        
        # Test rhythm features structure and types
        rhythm = features['rhythm']
        rhythm_keys = ['tempo', 'onset_density', 'syncopation_level', 'rhythmic_variance', 'beat_strength']
        for key in rhythm_keys:
            assert key in rhythm, f"Missing rhythm feature: {key}"
            assert isinstance(rhythm[key], (int, float, np.number))
            assert not np.isnan(rhythm[key])
        
        # Test harmony features structure and types
        harmony = features['harmony']
        harmony_keys = ['chroma_variance', 'key_strength', 'harmonic_change_rate', 'tonal_stability']
        for key in harmony_keys:
            assert key in harmony, f"Missing harmony feature: {key}"
            assert isinstance(harmony[key], (int, float, np.number))
            assert not np.isnan(harmony[key])
        
        # Test energy features structure and types
        energy = features['energy']
        energy_keys = ['energy_range', 'avg_energy', 'energy_trend', 'peak_density']
        for key in energy_keys:
            assert key in energy, f"Missing energy feature: {key}"
            assert isinstance(energy[key], (int, float, np.number))
            assert not np.isnan(energy[key])
        
        # Test spectral features structure and types
        spectral = features['spectral']
        spectral_keys = ['avg_brightness', 'brightness_variance', 'avg_rolloff', 'avg_bandwidth']
        for key in spectral_keys:
            assert key in spectral, f"Missing spectral feature: {key}"
            assert isinstance(spectral[key], (int, float, np.number))
            assert not np.isnan(spectral[key])
        
        # Test frequency features structure and types
        frequency = features['frequency']
        frequency_keys = ['low_proportion', 'mid_proportion', 'high_proportion', 'mid_low_ratio', 'high_mid_ratio']
        for key in frequency_keys:
            assert key in frequency, f"Missing frequency feature: {key}"
            assert isinstance(frequency[key], (int, float, np.number))
            assert not np.isnan(frequency[key])
        
        # Test proportions sum to ~1
        proportions = frequency['low_proportion'] + frequency['mid_proportion'] + frequency['high_proportion']
        assert 0.95 < proportions < 1.05, f"Frequency proportions should sum to ~1, got {proportions}"
    
    @patch('librosa.load')
    @patch('os.path.exists')
    def test_embedding_vector_shape_and_type(self, mock_exists, mock_load, service, mock_audio_data):
        """Test that create_embedding_vector returns correct shape and type"""
        # Setup mocks
        mock_exists.return_value = True
        mock_load.return_value = (mock_audio_data, 22050)
        
        # Load and extract features
        service.load_audio_file("fake_path.mp3")
        features = service.extract_global_features(max_duration=30)
        
        # Create embedding vector
        vector = service.create_embedding_vector(features)
        
        # Test vector properties
        assert isinstance(vector, np.ndarray)
        assert vector.shape == (19,), f"Expected shape (19,), got {vector.shape}"
        assert vector.dtype in [np.float64, np.float32], f"Expected float type, got {vector.dtype}"
        
        # Test values are finite and reasonable
        assert np.all(np.isfinite(vector)), "Vector contains non-finite values"
        assert np.all(vector >= 0), "Vector contains negative values (should be normalized to 0-1+ range)"
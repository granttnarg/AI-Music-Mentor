import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from unittest.mock import Mock, patch
from datetime import datetime

from db.models import UserUpload, Track, TrainingExample, Feedback


@pytest.fixture
def mock_db():
    """Mock database with realistic data"""
    db = Mock()
    session = Mock()
    db.get_session.return_value = session
    return db, session


@pytest.fixture
def mock_user_upload():
    """Mock user upload with realistic data"""
    upload = Mock(spec=UserUpload)
    upload.id = 1
    upload.input_track_id = 100
    upload.reference_track_id = 200
    upload.user_prompt = "Help me improve the drop section"
    upload.stage = "Half Finished"
    upload.genre = "techno"
    upload.session_id = "test-session-123"
    return upload


@pytest.fixture
def realistic_global_features():
    """Realistic global feature data based on actual training data"""
    return {
        "metadata": {"duration": 180.5, "sample_rate": 22050},
        "rhythm": {
            "tempo": 132.51,
            "onset_density": 6.13,
            "syncopation_level": 0.29,
            "rhythmic_variance": 0.013,
            "beat_strength": 0.60,
        },
        "harmony": {
            "chroma_variance": 0.011,
            "key_strength": 2.55,
            "harmonic_change_rate": 0.0007,
            "tonal_stability": 0.73,
        },
        "energy": {
            "energy_range": 0.36,
            "avg_energy": 0.35,
            "energy_trend": 9.47e-07,
            "peak_density": 12.01,
        },
        "spectral": {
            "avg_brightness": 1820.09,
            "brightness_variance": 445375.36,
            "avg_rolloff": 4141.78,
            "avg_bandwidth": 2424.82,
        },
        "frequency": {
            "low_proportion": 0.82,
            "mid_proportion": 0.14,
            "high_proportion": 0.025,
            "mid_low_ratio": 0.18,
            "high_mid_ratio": 0.17,
        },
    }


@pytest.fixture
def realistic_embedding():
    """Realistic 19-dimensional embedding based on training data"""
    return np.array(
        [
            0.66,
            0.40,
            0.29,
            0.13,
            0.11,
            0.85,
            0.15,
            0.73,
            0.36,
            0.35,
            0.0009,
            0.48,
            0.22,
            0.22,
            0.82,
            0.14,
            0.025,
            0.18,
            0.17,
        ]
    )


@pytest.fixture
def mock_input_track(realistic_global_features, realistic_embedding):
    """Mock input track with realistic embedding and features"""
    track = Mock(spec=Track)
    track.id = 100
    track.file_path = "/path/to/input--test_track.wav"
    track.duration = 180.5
    track.sample_rate = 22050
    track.global_embedding = realistic_embedding
    track.smoothed_arrangement_pattern = "O-A-O"
    track.raw_arrangement_pattern = "1A-1O-2A-9O"
    track.global_feature_data = realistic_global_features.copy()
    return track


@pytest.fixture
def mock_reference_track(realistic_global_features, realistic_embedding):
    """Mock reference track with slightly different features"""
    track = Mock(spec=Track)
    track.id = 200
    track.file_path = "/path/to/ref--reference_track.mp3"
    track.duration = 331.08
    track.sample_rate = 22050
    # Slightly different embedding
    track.global_embedding = realistic_embedding + np.random.normal(0, 0.1, 19)
    track.smoothed_arrangement_pattern = "intro-buildup-drop-breakdown-outro"
    track.raw_arrangement_pattern = "A-B-C-B-D"

    # Modify features slightly for reference
    ref_features = realistic_global_features.copy()
    ref_features["rhythm"]["tempo"] = 128.0
    ref_features["energy"]["avg_energy"] = 0.42
    track.global_feature_data = ref_features
    return track


@pytest.fixture
def mock_similar_tracks(realistic_global_features, realistic_embedding):
    """Mock similar tracks with training examples"""
    tracks = []
    for i in range(3):
        track = Mock(spec=Track)
        track.id = 300 + i
        track.file_path = f"data/training/track_{i}--example.wav"
        track.duration = 180.0 + i * 10
        track.sample_rate = 22050
        # Slightly varied embeddings
        noise = np.random.normal(0, 0.05, 19)
        track.global_embedding = realistic_embedding + noise
        track.smoothed_arrangement_pattern = ["O-A-O", "intro-drop-outro", "A-B-A"][i]

        # Vary features slightly for each track
        features = realistic_global_features.copy()
        features["rhythm"]["tempo"] = 132.5 + i * 2
        features["energy"]["avg_energy"] = 0.35 + i * 0.05
        track.global_feature_data = features
        tracks.append(track)
    return tracks


@pytest.fixture
def mock_training_examples(mock_similar_tracks):
    """Mock training examples for the similar tracks"""
    examples = []
    for i, track in enumerate(mock_similar_tracks):
        example = Mock(spec=TrainingExample)
        example.id = 400 + i
        example.example_track_id = track.id
        example.reference_track_id = 500 + i  # Different ref track
        example.genre = "techno"
        example.stage = "Half Finished"
        example.created_at = datetime.now()
        examples.append(example)
    return examples


@pytest.fixture
def mock_feedback_items(mock_training_examples):
    """Mock feedback items for training examples with realistic content"""
    all_feedback = []
    feedback_data = [
        (
            "rhythm",
            "The kick drum pattern could use more variation. Try adding some subtle timing shifts to create groove.",
        ),
        (
            "eq",
            "High-pass filter the bass elements around 80Hz to clean up the low end mud.",
        ),
        (
            "arrangement",
            "The drop section needs more energy buildup. Consider adding risers and tension elements.",
        ),
        (
            "energy",
            "The energy curve feels flat. Try automating the filter cutoff to create more movement.",
        ),
        (
            "global",
            "Overall mix feels compressed. Give more headroom to the transients.",
        ),
    ]

    for i, example in enumerate(mock_training_examples):
        # Add 1-2 feedback items per example
        for j in range(1 + i % 2):  # 1-2 items per example
            feedback_type, feedback_text = feedback_data[(i + j) % len(feedback_data)]
            feedback = Mock(spec=Feedback)
            feedback.id = 600 + i * 10 + j
            feedback.training_example_id = example.id
            feedback.feedback_type = feedback_type
            feedback.feedback_text = feedback_text
            feedback.created_at = datetime.now()
            all_feedback.append(feedback)

    return all_feedback


@pytest.fixture
def mock_rag_operations(mock_similar_tracks, mock_db):
    """Mock RAG operations"""
    operations = Mock()
    operations.find_similar_tracks_with_training_examples.return_value = (
        mock_similar_tracks
    )
    # Add db attribute for backward compatibility
    db, _ = mock_db
    operations.db = db
    return operations


@pytest.fixture
def mock_rag_prompts():
    """Mock prompts for RAG system"""
    return {
        "feedback_generation": {
            "template": """You are an AI music mentor.

Context: {examples}

Input Track Pattern: {input_pattern}
Reference Track Pattern: {reference_pattern}
Feature Comparison: {feature_comparison}
Genre: {genre}
Stage: {stage}

Question: {question}

Provide detailed, actionable feedback based on the similar examples."""
        }
    }


@pytest.fixture
def setup_rag(mock_db, mock_rag_operations, mock_rag_prompts):
    """Set up AudioRAG with mocked dependencies"""
    from services.audio_rag import AudioRAG

    db, session = mock_db

    with patch.dict(os.environ, {"DEVELOPMENT_BASE_URL": "http://localhost:11434"}):
        with patch(
            "services.audio_rag.PromptLoader._load_prompts",
            return_value=mock_rag_prompts,
        ):
            with patch("services.audio_rag.AudioRAGOperations"):
                with patch("services.audio_rag.ChatPromptTemplate"):
                    with patch("services.audio_rag.ChatOllama"):
                        with patch("services.audio_rag.StrOutputParser"):
                            # Create AudioRAG instance with injected operations and prompts
                            mock_chain = Mock()
                            mock_chain.invoke.return_value = "Here's some great feedback about your techno track based on similar examples!"
                            rag = AudioRAG(
                                mock_rag_operations, mock_rag_prompts, mock_chain
                            )

                            # Chain is already injected above

                            return rag, session


@pytest.fixture
def realistic_similar_examples():
    """Pre-built similar examples structure for testing formatters"""
    return [
        {
            "training_example_id": 400,
            "similarity_rank": 1,
            "example_track": {
                "id": 300,
                "file_path": "data/training/track_0--example.wav",
                "arrangement_pattern": "O-A-O",
            },
            "feedback": [
                {
                    "type": "rhythm",
                    "text": "The kick drum pattern could use more variation. Try adding some subtle timing shifts.",
                },
                {
                    "type": "eq",
                    "text": "High-pass filter the bass elements around 80Hz to clean up the low end.",
                },
            ],
        }
    ]

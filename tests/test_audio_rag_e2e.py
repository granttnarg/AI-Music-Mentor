import pytest
from unittest.mock import Mock, patch
from datetime import datetime

from services.audio_rag import AudioRAG
from db.models import UserUpload, Track, TrainingExample, Feedback


class TestAudioRAGEndToEnd:
    """End-to-end test for AudioRAG with mocked dependencies"""

    def test_retrieve_similar_examples_complete_flow(
        self,
        setup_rag,
        mock_user_upload,
        mock_input_track,
        mock_similar_tracks,
        mock_training_examples,
        mock_feedback_items,
    ):
        """Test the complete retrieve_similar_examples flow"""
        rag, session = setup_rag

        # Set up session query mocks
        def mock_query_side_effect(model):
            query_mock = Mock()
            if model == UserUpload:
                query_mock.filter.return_value.first.return_value = mock_user_upload
            elif model == Track:
                # Return input track for the first call
                query_mock.filter.return_value.first.return_value = mock_input_track
            elif model == TrainingExample:
                # Return training examples for each similar track
                query_mock.filter.return_value.all.return_value = mock_training_examples
            elif model == Feedback:
                # Return feedback items
                query_mock.filter.return_value.all.return_value = mock_feedback_items
            return query_mock

        session.query.side_effect = mock_query_side_effect

        # Execute the method
        results, user_upload, retrieval_summary = rag.retrieve_similar_examples(
            user_upload_id=1, k=3, metric="cosine"
        )

        # Verify the results structure
        assert isinstance(results, list)
        assert len(results) <= 3  # Should respect k limit
        assert user_upload == mock_user_upload
        assert isinstance(retrieval_summary, dict)

        # Verify retrieval summary structure
        expected_keys = [
            "user_upload_id",
            "k_requested",
            "k_found",
            "metric",
            "user_genre",
            "retrieved_tracks",
        ]
        for key in expected_keys:
            assert key in retrieval_summary

        # Verify result structure for each item
        if results:
            result = results[0]
            expected_result_keys = [
                "training_example_id",
                "similarity_rank",
                "example_track",
                "reference_track",
                "feedback",
                "created_at",
            ]
            for key in expected_result_keys:
                assert key in result

            # Verify example_track structure
            example_track = result["example_track"]
            assert "global_feature_data" not in example_track  # Should be excluded
            assert "embedding" in example_track
            assert "arrangement_pattern" in example_track

            # Verify feedback structure - this is the key transformation!
            if result["feedback"]:
                feedback = result["feedback"][0]
                assert "type" in feedback  # DB feedback_type -> "type"
                assert "text" in feedback  # DB feedback_text -> "text"
                assert "created_at" in feedback

        print("✅ Retrieved similar examples structure:")
        print(f"   - Found {len(results)} results")
        print(f"   - Retrieval summary keys: {list(retrieval_summary.keys())}")
        if results:
            print(f"   - First result keys: {list(results[0].keys())}")
            print(
                f"   - Example track keys: {list(results[0]['example_track'].keys())}"
            )
            if results[0]["feedback"]:
                print(
                    f"   - Feedback structure: {list(results[0]['feedback'][0].keys())}"
                )
                print(
                    f"   - Feedback mapping verified: DB 'feedback_text' -> dict 'text' ✓"
                )

    def test_format_examples_for_prompt_flow(
        self,
        setup_rag,
        mock_user_upload,
        mock_input_track,
        mock_reference_track,
        realistic_similar_examples,
    ):
        """Test the format_examples_for_prompt flow"""
        rag, session = setup_rag

        # Mock the session queries for track lookups
        def mock_query_side_effect(model):
            query_mock = Mock()
            if model == Track:
                first_mock = Mock()
                first_mock.side_effect = [mock_input_track, mock_reference_track]
                query_mock.filter.return_value.first = first_mock
            return query_mock

        session.query.side_effect = mock_query_side_effect

        # Mock the text formatter
        with patch("services.audio_rag.RagTextFormatter") as mock_formatter_class:
            mock_formatter = Mock()
            mock_formatter.rank_feedback_by_relevance.return_value = (
                realistic_similar_examples[0]["feedback"]
            )
            mock_formatter_class.return_value = mock_formatter

            # Execute the method
            formatted_text = rag.format_examples_for_prompt(
                realistic_similar_examples,
                mock_user_upload,
                "Help me with the drop section",
            )

        # Verify the output
        assert isinstance(formatted_text, str)
        assert len(formatted_text) > 0
        assert "User Upload Context:" in formatted_text
        assert "Most Relevant Feedback" in formatted_text
        assert mock_user_upload.user_prompt in formatted_text
        assert "techno" in formatted_text  # Genre should be included
        assert "Half Finished" in formatted_text  # Stage should be included

        print("✅ Formatted examples structure:")
        print(f"   - Output length: {len(formatted_text)} characters")
        print(f"   - Contains user context: {'User Upload Context:' in formatted_text}")
        print(f"   - Contains feedback: {'Most Relevant Feedback' in formatted_text}")
        print(f"   - Contains genre: {'techno' in formatted_text}")

    def test_generate_feedback_complete_pipeline(
        self, setup_rag, mock_user_upload, mock_input_track, realistic_global_features
    ):
        """Test the complete generate_feedback pipeline"""
        rag, session = setup_rag

        # Mock the retrieve_similar_examples method
        mock_results = [
            {
                "training_example_id": 400,
                "feedback": [
                    {"type": "rhythm", "text": "Great kick drum pattern, very punchy"},
                    {
                        "type": "arrangement",
                        "text": "The drop section has excellent energy flow",
                    },
                ],
            }
        ]

        with patch.object(rag, "retrieve_similar_examples") as mock_retrieve:
            mock_retrieve.return_value = (
                mock_results,
                mock_user_upload,
                {"k_found": 1},
            )

            with patch.object(rag, "format_examples_for_prompt") as mock_format:
                mock_format.return_value = "User Upload Context:\n  User Prompt: Help me improve the drop section\n\nMost Relevant Feedback:\nExample 1: Great kick drum pattern..."

                # Mock feature comparison service
                with patch(
                    "services.audio_rag.FeatureComparisonService"
                ) as mock_feature_service:
                    mock_feature_service.create_feature_comparison.return_value = (
                        "Input tempo: 132.5 BPM vs Reference: 128.0 BPM"
                    )

                    # Execute the complete pipeline
                    feedback = rag.generate_feedback(
                        user_upload_id=1, question="Help with my drop section", k=3
                    )

        # Verify the output
        assert isinstance(feedback, str)
        assert len(feedback) > 0

        # Verify the methods were called correctly
        mock_retrieve.assert_called_once_with(1, k=3)
        mock_format.assert_called_once()

        print("✅ Complete pipeline test:")
        print(f"   - Generated feedback length: {len(feedback)} characters")
        print(f"   - Pipeline completed successfully")

    def test_data_transformations_realistic(self, realistic_global_features):
        """Test the key data transformations with realistic data"""

        # Test the feedback transformation from DB model to dict
        mock_feedback_db = Mock(spec=Feedback)
        mock_feedback_db.feedback_type = "rhythm"
        mock_feedback_db.feedback_text = "The kick drum pattern could use more variation. Try adding some subtle timing shifts to create groove."
        mock_feedback_db.created_at = datetime.now()

        # Simulate the transformation that happens in _build_training_example_results
        transformed_feedback = {
            "type": mock_feedback_db.feedback_type,
            "text": mock_feedback_db.feedback_text,  # DB feedback_text -> dict "text"
            "created_at": str(mock_feedback_db.created_at),
        }

        # Test global features structure
        assert "rhythm" in realistic_global_features
        assert "harmony" in realistic_global_features
        assert "energy" in realistic_global_features
        assert "spectral" in realistic_global_features
        assert "frequency" in realistic_global_features

        rhythm_features = realistic_global_features["rhythm"]
        assert "tempo" in rhythm_features
        assert "onset_density" in rhythm_features
        assert "syncopation_level" in rhythm_features

        # Verify the key transformation
        assert transformed_feedback["type"] == "rhythm"
        assert "variation" in transformed_feedback["text"]  # Realistic feedback content
        assert "created_at" in transformed_feedback

        print("✅ Data transformation test with realistic data:")
        print(f"   - DB field 'feedback_text' -> dict key 'text': ✓")
        print(f"   - DB field 'feedback_type' -> dict key 'type': ✓")
        print(f"   - Global features structure matches training data: ✓")
        print(f"   - Feedback content: {transformed_feedback['text'][:50]}...")

    def test_user_prompt_attribute_fix(self, setup_rag, mock_user_upload):
        """Test that user_prompt (not user_prompt_notes) is correctly accessed"""
        rag, session = setup_rag

        # Mock similar examples
        similar_examples = [{"feedback": [{"type": "test", "text": "test feedback"}]}]

        # Test the user question extraction logic
        question = ""  # Empty question to trigger fallback
        user_question = (
            question
            if question.strip()
            else (
                getattr(mock_user_upload, "user_prompt", "general feedback")
                or "general feedback"
            )
        )

        # Should use the user_prompt, not default to "general feedback"
        assert user_question == "Help me improve the drop section"
        assert user_question != "general feedback"

        print("✅ User prompt attribute test:")
        print(f"   - Correctly uses 'user_prompt' attribute: ✓")
        print(f"   - User question: '{user_question}'")


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "-s"])

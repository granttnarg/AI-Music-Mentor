from typing import List, Dict, Any
from db.operations import AudioRAGOperations
from db.db import AudioRAGDatabase
from db.models import TrainingExample, Track, UserUpload, Feedback
import os


class AudioRAG:
    def __init__(self, db: AudioRAGDatabase, openai_api_key: str = None):
        self.db = db
        self.operations = AudioRAGOperations(db)

    def retrieve_similar_examples(self, user_upload_id: int, k: int = 5, metric: str = "cosine") -> List[Dict[str, Any]]:
        """
        Retrieve top-k most similar training examples for a given user upload

        Args:
            user_upload_id: ID of the user upload
            k: Number of similar examples to return
            metric: Distance metric ("cosine" or "euclidean")

        Returns:
            List of dictionaries containing training example data and similarity info
        """
        session = self.db.get_session()

        try:
            # Get the user upload and its input track
            user_upload = session.query(UserUpload).filter(UserUpload.id == user_upload_id).first()
            if not user_upload:
                raise ValueError(f"User upload {user_upload_id} not found")

            input_track = session.query(Track).filter(Track.id == user_upload.input_track_id).first()
            if not input_track or input_track.global_embedding is None:
                raise ValueError(f"Input track embedding not found for user upload {user_upload_id}")

            # Use existing find_similar_tracks method to get similar tracks
            similar_tracks = self.operations.find_similar_tracks(
                embedding=list(input_track.global_embedding),
                metric=metric,
                limit=k * 3  # Get more tracks since we'll filter for training examples
            )

            # Filter tracks that are part of training examples and get the training data
            results = []
            for track in similar_tracks:
                if len(results) >= k:
                    break

                # Find training examples where this track is the example track
                training_examples = session.query(TrainingExample).filter(
                    TrainingExample.example_track_id == track.id
                ).all()

                for training_example in training_examples:
                    if len(results) >= k:
                        break

                    # Get reference track
                    reference_track = session.query(Track).filter(
                        Track.id == training_example.reference_track_id
                    ).first()

                    # Get feedback for this training example
                    feedback_items = session.query(Feedback).filter(
                        Feedback.training_example_id == training_example.id
                    ).all()

                    result = {
                        "training_example_id": training_example.id,
                        "similarity_rank": len(results) + 1,
                        "example_track": {
                            "id": track.id,
                            "file_path": track.file_path,
                            "embedding": list(track.global_embedding),
                            "duration": track.duration,
                            "sample_rate": track.sample_rate
                        },
                        "reference_track": {
                            "id": reference_track.id,
                            "file_path": reference_track.file_path,
                            "embedding": list(reference_track.global_embedding) if reference_track.global_embedding is not None else None,
                            "duration": reference_track.duration,
                            "sample_rate": reference_track.sample_rate
                        } if reference_track else None,
                        "feedback": [
                            {
                                "type": fb.feedback_type,
                                "text": fb.feedback_text,
                                "created_at": str(fb.created_at)
                            } for fb in feedback_items
                        ],
                        "created_at": str(training_example.created_at)
                    }
                    results.append(result)

            return results

        except Exception as e:
            print(f"Error retrieving similar examples: {e}")
            raise
        finally:
            session.close()


# Example usage
if __name__ == "__main__":
    from dotenv import load_dotenv
    from db.db import AudioRAGDatabase

    # Load environment variables from .env file
    load_dotenv()

    # Initialize database and RAG
    connection_url = os.getenv('DB_CONNECTION_URL', 'postgresql://postgres:<ADD_TOENV_FILE>')
    db = AudioRAGDatabase(connection_url)
    rag = AudioRAG(db)

    # Test with user upload ID 1
    try:
        similar_examples = rag.retrieve_similar_examples(user_upload_id=1, k=3)

        print(f"Found {len(similar_examples)} similar training examples:")
        for i, example in enumerate(similar_examples, 1):
            print(f"\n--- Similar Example {i} ---")
            print(f"Training Example ID: {example['training_example_id']}")
            print(f"Example Track: {example['example_track']['file_path']}")
            if example['reference_track']:
                print(f"Reference Track: {example['reference_track']['file_path']}")

            if example['feedback']:
                print("Feedback:")
                for fb in example['feedback']:
                    print(f"  - {fb['type']}: {fb['text']}")
            else:
                print("No feedback available")

    except Exception as e:
        print(f"Error: {e}")
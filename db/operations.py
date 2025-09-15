from .db import AudioRAGDatabase
from .models import Track, UserUpload, Feedback, TrainingExample
from datetime import datetime
from typing import List


class AudioRAGOperations:
    def __init__(self, db: AudioRAGDatabase):
        self.db = db

    def get_track(self, track_id: int):
        """Get a track by ID"""
        session = self.db.get_session()

        try:
            track = session.query(Track).filter(Track.id == track_id).first()
            if track:
                return {
                    "id": track.id,
                    "file_path": track.file_path,
                    "duration": track.duration,
                    "sample_rate": track.sample_rate,
                    "embedding": track.global_embedding,
                    "processed_at": track.processed_at,
                }
            else:
                return None

        except Exception as e:
            print(f"Error getting track {track_id}: {e}")
            raise
        finally:
            session.close()

    def get_track_by_file_path(self, file_path: str) -> bool:
        """Check if a track with this file path already exists"""
        session = self.db.get_session()
        try:
            existing = session.query(Track).filter(Track.file_path == file_path).first()
            return existing is not None
        finally:
            session.close()

    def add_user_upload(
        self,
        input_track_path: str,
        ref_track_path: str,
        input_duration: float,
        input_sample_rate,
        input_embedding,
        ref_duration: float,
        ref_sample_rate,
        ref_embedding,
        user_prompt,
        stage,
        genre,
        session_id,
        input_file_size_bytes,
        reference_file_size_bytes,
        input_original_filename,
        reference_original_filename,
    ):
        session = self.db.get_session()

        try:
            input_track = self._add_track(
                session,
                input_track_path,
                input_duration,
                input_sample_rate,
                input_embedding,
            )
            ref_track = self._add_track(
                session,
                ref_track_path,
                ref_duration,
                ref_sample_rate,
                ref_embedding,
            )
            session.flush()  # This should be enough to get the ID

            print(f"Got track ID: {input_track.id}")  # Debug line

            upload = UserUpload(
                input_track_id=input_track.id,
                reference_track_id=ref_track.id,
                user_prompt=user_prompt,
                stage=stage,
                genre=genre,
                session_id=session_id,
                input_file_size_bytes=input_file_size_bytes,
                reference_file_size_bytes=reference_file_size_bytes,
                input_original_filename=input_original_filename,
                reference_original_filename=reference_original_filename,
            )
            session.add(upload)
            session.commit()

            upload_id = upload.id
            return upload_id

        except Exception as e:
            session.rollback()
            print(f"Error: {e}")
            raise
        finally:
            session.close()

    def add_training_example(
        self,
        input_track_path: str,
        ref_track_path: str,
        input_duration: float,
        input_sample_rate: int,
        input_embedding: List[float],
        ref_duration: float,
        ref_sample_rate: int,
        ref_embedding: List[float],
        feedback_items: List[dict],
        genre: str = "techno",
    ):
        """
        Add a training example with tracks and feedback to the database.

        Args:
            input_track_path: Path to saved input track file
            ref_track_path: Path to saved reference track file
            input_duration, input_sample_rate, input_embedding: Input track features
            ref_duration, ref_sample_rate, ref_embedding: Reference track features
            feedback_items: List of dicts with 'feedback_type' and 'feedback_text'

        Returns:
            int: The training example ID
        """
        session = self.db.get_session()

        try:
            # Create track records
            input_track = self._add_track(
                session,
                input_track_path,
                input_duration,
                input_sample_rate,
                input_embedding,
            )

            ref_track = self._add_track(
                session,
                ref_track_path,
                ref_duration,
                ref_sample_rate,
                ref_embedding,
            )

            session.flush()  # Get track IDs

            # Create training example
            training_example = TrainingExample(
                example_track_id=input_track.id,
                reference_track_id=ref_track.id,
                genre=genre,
            )
            session.add(training_example)
            session.flush()  # Get training example ID

            # Add feedback items
            for feedback_item in feedback_items:
                feedback = Feedback(
                    training_example_id=training_example.id,
                    feedback_type=feedback_item["feedback_type"],
                    feedback_text=feedback_item["feedback_text"],
                )
                session.add(feedback)

            session.commit()
            return training_example.id

        except Exception as e:
            session.rollback()
            print(f"Error adding training example: {e}")
            raise
        finally:
            session.close()

    def get_all_training_examples(self):
        """Get all training examples with track and feedback information."""
        session = self.db.get_session()
        try:
            examples = (
                session.query(TrainingExample)
                .order_by(TrainingExample.created_at.desc())
                .all()
            )

            result = []
            for example in examples:
                # Get feedback items
                feedback_items = (
                    session.query(Feedback)
                    .filter(Feedback.training_example_id == example.id)
                    .all()
                )

                result.append(
                    {
                        "id": example.id,
                        "genre": example.genre,
                        "created_at": example.created_at,
                        "input_track": {
                            "id": example.example_track.id,
                            "file_path": example.example_track.file_path,
                            "duration": example.example_track.duration,
                        },
                        "reference_track": {
                            "id": example.reference_track.id,
                            "file_path": example.reference_track.file_path,
                            "duration": example.reference_track.duration,
                        },
                        "feedback_items": [
                            {
                                "id": fb.id,
                                "type": fb.feedback_type,
                                "text": fb.feedback_text,
                                "created_at": fb.created_at,
                            }
                            for fb in feedback_items
                        ],
                    }
                )

            return result

        except Exception as e:
            print(f"Error getting training examples: {e}")
            raise
        finally:
            session.close()

    def get_training_example_by_id(self, training_id: int):
        """Get a specific training example by ID."""
        session = self.db.get_session()
        try:
            example = (
                session.query(TrainingExample)
                .filter(TrainingExample.id == training_id)
                .first()
            )
            if not example:
                return None

            # Get feedback items
            feedback_items = (
                session.query(Feedback)
                .filter(Feedback.training_example_id == example.id)
                .all()
            )

            return {
                "id": example.id,
                "genre": example.genre,
                "created_at": example.created_at,
                "input_track": {
                    "id": example.example_track.id,
                    "file_path": example.example_track.file_path,
                    "duration": example.example_track.duration,
                },
                "reference_track": {
                    "id": example.reference_track.id,
                    "file_path": example.reference_track.file_path,
                    "duration": example.reference_track.duration,
                },
                "feedback_items": [
                    {
                        "id": fb.id,
                        "type": fb.feedback_type,
                        "text": fb.feedback_text,
                        "created_at": fb.created_at,
                    }
                    for fb in feedback_items
                ],
            }

        except Exception as e:
            print(f"Error getting training example {training_id}: {e}")
            raise
        finally:
            session.close()

    def update_training_example_feedback(
        self, training_id: int, feedback_updates: list, genre: str | None = None
    ):
        """Update feedback items for a training example."""
        session = self.db.get_session()
        try:
            # Get training example
            example = (
                session.query(TrainingExample)
                .filter(TrainingExample.id == training_id)
                .first()
            )
            if not example:
                raise ValueError(f"Training example {training_id} not found")

            # Update genre if provided
            if genre:
                setattr(example, "genre", genre)

            # First, delete all existing feedback (we'll re-add what we want to keep)
            session.query(Feedback).filter(
                Feedback.training_example_id == training_id
            ).delete()

            # Add all feedback items (this includes both updates and new items)
            for fb_update in feedback_updates:
                new_feedback = Feedback(
                    training_example_id=training_id,
                    feedback_type=fb_update["type"],
                    feedback_text=fb_update["text"],
                )
                session.add(new_feedback)

            session.commit()
            return training_id

        except Exception as e:
            session.rollback()
            print(f"Error updating training example {training_id}: {e}")
            raise
        finally:
            session.close()

    def find_similar_tracks(
        self,
        embedding: List[float],
        metric: str = "cosine",
        limit: int = 5,
        threshold: float | None = None,
    ) -> List[Track]:
        """Find tracks using specified distance metric"""
        session = self.db.get_session()

        try:
            if metric == "cosine":
                distance = Track.global_embedding.cosine_distance(embedding)
                query = session.query(Track).order_by(distance)
                if threshold is not None:
                    query = query.filter(distance <= threshold)

            elif metric == "euclidean":
                distance = Track.global_embedding.l2_distance(embedding)
                query = session.query(Track).order_by(distance)
                if threshold is not None:
                    query = query.filter(distance <= threshold)

            elif metric == "inner_product":
                score = Track.global_embedding.max_inner_product(embedding)
                query = session.query(Track).order_by(score.desc())
                if threshold is not None:
                    query = query.filter(score >= threshold)
            else:
                raise ValueError(f"Unknown metric: {metric}")

            return query.limit(limit).all()

        except Exception as e:
            print(f"✗ Error finding similar tracks ({metric}): {e}")
            raise
        finally:
            session.close()

    ## PRIVATE METHODS ##

    def _add_track(
        self,
        session,
        file_path: str,
        duration: float,
        sample_rate: int,
        embedding: List[float],
    ) -> Track:
        """Add a track using the provided session"""

        # Check if track already exists
        existing_track = (
            session.query(Track).filter(Track.file_path == file_path).first()
        )

        if existing_track:
            # Update existing track
            existing_track.duration = duration
            existing_track.sample_rate = sample_rate
            existing_track.global_embedding = embedding
            existing_track.processed_at = datetime.now()
            return existing_track
        else:
            # Create new track
            track = Track(
                file_path=file_path,
                duration=duration,
                sample_rate=sample_rate,
                global_embedding=embedding,
                processed_at=datetime.now(),
            )
            session.add(track)
            return track

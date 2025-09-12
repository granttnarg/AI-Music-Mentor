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
        user_prompt,
        stage,
        genre,
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
            ref_track = self._add_track(session, ref_track_path, input_duration, input_sample_rate, input_embedding)
            session.flush()  # This should be enough to get the ID


            print(f"Got track ID: {input_track.id}")  # Debug line

            upload = UserUpload(
                input_track_id=input_track.id,
                reference_track_id=ref_track.id,
                user_prompt=user_prompt,
                stage=stage,
                genre=genre,
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

    # Currently we make a training example using tracks in our DB, this is not how it should work for the final app.
    # We will want to create tracks from the uploads at the same time as the training example and feedback
    # However this helps us get some examples in our DB to test out the RAG flow we will come back to this.
    # TODO: ADJUST TO CREATE TRACKS ON THE FLY
    def add_training_example_for_mockdata(self, example_track_path, reference_track_path, feedback_items):
        """
        Add a training example with feedback.

        Args:
            example_track_path (str): Path to the example track
            reference_track_path (str): Path to the reference track
            feedback_items (list): List of dicts with 'feedback_type' and 'feedback_text'
                                e.g. [{'feedback_type': 'rhythm', 'feedback_text': 'Too slow'}]

        Returns:
            int: The training example ID
        """
        session = self.db.get_session()
        try:
            # Get or create example track
            example_track = session.query(Track).filter_by(file_path=example_track_path).first()
            if not example_track:
                raise ValueError(f"Example track not found: {example_track_path}")

            # Get or create reference track
            reference_track = session.query(Track).filter_by(file_path=reference_track_path).first()
            if not reference_track:
                raise ValueError(f"Reference track not found: {reference_track_path}")

            # Create training example
            training_example = TrainingExample(
                example_track_id=example_track.id,
                reference_track_id=reference_track.id
            )
            session.add(training_example)
            session.flush()  # Get the ID

            # Add feedback items
            for feedback_item in feedback_items:
                feedback = Feedback(
                    training_example_id=training_example.id,
                    feedback_type=feedback_item['feedback_type'],
                    feedback_text=feedback_item['feedback_text']
                )
                session.add(feedback)

            session.commit()
            return training_example.id

        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def find_similar_tracks(
        self,
        embedding: List[float],
        metric: str = "cosine",
        limit: int = 5,
        threshold: float = None,
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

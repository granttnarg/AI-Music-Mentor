from .db import AudioRAGDatabase
from .models import Track, UserUpload, Feedback, TrainingExample
from datetime import datetime
from typing import List, Dict, Optional, Union
import json
from pathlib import Path


class AudioRAGOperations:
    def __init__(self, db: AudioRAGDatabase):
        self.db = db
        self.arrangement_classifier = None  # Lazy load
        self._pending_arrangement_analysis = []  # Queue for async processing

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
                    "raw_arrangement_pattern": track.raw_arrangement_pattern,
                    "smoothed_arrangement_pattern": track.smoothed_arrangement_pattern,
                    "raw_predictions": track.raw_predictions,
                    "raw_confidence_scores": track.raw_confidence_scores,
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

    def _get_arrangement_classifier(self):
        """Lazy load the arrangement classifier"""
        if self.arrangement_classifier is None:
            try:
                # Import here to avoid circular imports and load only when needed
                import sys
                import os
                from pathlib import Path

                # Add project root to path for classifier imports
                project_root = Path(__file__).parent.parent
                sys.path.append(str(project_root / "src"))

                from classifier.arrangement_classifier import ArrangementClassifier

                model_dir = (
                    project_root / "models" / "arrangement_classifier" / "4classes"
                )
                self.arrangement_classifier = ArrangementClassifier(str(model_dir))

                if not self.arrangement_classifier.load_model():
                    print("⚠️ Failed to load arrangement classifier model")
                    self.arrangement_classifier = None
                else:
                    print("✅ Arrangement classifier loaded successfully")

            except Exception as e:
                print(f"⚠️ Error loading arrangement classifier: {e}")
                self.arrangement_classifier = None

        return self.arrangement_classifier

    def _classify_track_arrangement(
        self, file_path: str
    ) -> Dict[str, Union[str, None]]:
        """Private method to analyze track arrangement using new data structure"""
        try:
            classifier = self._get_arrangement_classifier()
            if classifier is None:
                return {
                    "raw_pattern": None,
                    "smoothed_pattern": None,
                    "raw_predictions": None,
                    "raw_confidence_scores": None,
                }

            print(f"🎵 Analyzing arrangement for: {Path(file_path).name}")

            # Use the updated analyze_arrangement_structure method
            arrangement_data = classifier.analyze_arrangement_structure(
                file_path, min_segment_length=2, confidence_threshold=0.4
            )

            if not arrangement_data:
                return {
                    "raw_pattern": None,
                    "smoothed_pattern": None,
                    "raw_predictions": None,
                    "raw_confidence_scores": None,
                }

            print(f"✅ Arrangement analysis complete:")
            print(f"   Raw: {arrangement_data['raw_pattern']}")
            print(f"   Smoothed: {arrangement_data['smoothed_pattern']}")

            return {
                "raw_pattern": arrangement_data["raw_pattern"],
                "smoothed_pattern": arrangement_data["smoothed_pattern"],
                "raw_predictions": json.dumps(arrangement_data["raw_predictions"]),
                "raw_confidence_scores": json.dumps(
                    arrangement_data["raw_confidence_scores"]
                ),
            }

        except Exception as e:
            print(f"⚠️ Arrangement analysis failed for {file_path}: {e}")
            return {
                "raw_pattern": None,
                "smoothed_pattern": None,
                "raw_predictions": None,
                "raw_confidence_scores": None,
            }

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
                classify_arrangement=True,
                sync_arrangement=True,  # User uploads need immediate arrangement data for RAG
            )
            ref_track = self._add_track(
                session,
                ref_track_path,
                ref_duration,
                ref_sample_rate,
                ref_embedding,
                classify_arrangement=True,
                sync_arrangement=True,  # User uploads need immediate arrangement data for RAG
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
        classify_arrangement: bool = True,
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
                classify_arrangement,
                sync_arrangement=False,  # Training examples use async processing for performance
            )

            ref_track = self._add_track(
                session,
                ref_track_path,
                ref_duration,
                ref_sample_rate,
                ref_embedding,
                classify_arrangement,
                sync_arrangement=False,  # Training examples use async processing for performance
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
                            "raw_arrangement_pattern": example.example_track.raw_arrangement_pattern,
                            "smoothed_arrangement_pattern": example.example_track.smoothed_arrangement_pattern,
                            "raw_predictions": example.example_track.raw_predictions,
                            "raw_confidence_scores": example.example_track.raw_confidence_scores,
                        },
                        "reference_track": {
                            "id": example.reference_track.id,
                            "file_path": example.reference_track.file_path,
                            "duration": example.reference_track.duration,
                            "raw_arrangement_pattern": example.reference_track.raw_arrangement_pattern,
                            "smoothed_arrangement_pattern": example.reference_track.smoothed_arrangement_pattern,
                            "raw_predictions": example.reference_track.raw_predictions,
                            "raw_confidence_scores": example.reference_track.raw_confidence_scores,
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
        classify_arrangement: bool = True,
        sync_arrangement: bool = False,
    ) -> Track:
        """Add a track using the provided session
        
        Args:
            sync_arrangement: If True, process arrangement analysis synchronously (blocking).
                             If False, queue for async processing. Use True for user uploads
                             that need immediate arrangement data for RAG feedback.
        """

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
            
            # If sync arrangement requested and track doesn't have arrangement data, process it
            if classify_arrangement and sync_arrangement and not existing_track.raw_arrangement_pattern:
                session.commit()  # Commit updates first
                self.update_track_arrangement(existing_track.id, file_path)
                print(f"✅ Added missing arrangement data to existing track {existing_track.id}")

            return existing_track
        else:
            # Create new track WITHOUT arrangement data first (fast)
            track = Track(
                file_path=file_path,
                duration=duration,
                sample_rate=sample_rate,
                global_embedding=embedding,
                processed_at=datetime.now(),
                raw_arrangement_pattern=None,  # Will be updated later
                raw_predictions=None,
                raw_confidence_scores=None,
                smoothed_arrangement_pattern=None,
            )
            session.add(track)
            session.flush()  # Get the track ID

            # If arrangement classification requested
            if classify_arrangement:
                if sync_arrangement:
                    # Process arrangement synchronously (blocking)
                    session.commit()  # Commit track first so we can update it
                    self.update_track_arrangement(track.id, file_path)
                    print(f"✅ Synchronous arrangement analysis complete for track {track.id}")
                else:
                    # Store track ID for async processing
                    self._pending_arrangement_analysis.append(
                        {"track_id": track.id, "file_path": file_path}
                    )

            return track

    def process_pending_arrangements(self):
        """Process all pending arrangement analysis synchronously"""
        if not self._pending_arrangement_analysis:
            return

        print(
            f"🎵 Processing {len(self._pending_arrangement_analysis)} pending arrangement analyses..."
        )

        for item in self._pending_arrangement_analysis:
            try:
                self.update_track_arrangement(item["track_id"], item["file_path"])
                print(f"✅ Updated arrangement for track {item['track_id']}")
            except Exception as e:
                print(
                    f"❌ Failed to analyze arrangement for track {item['track_id']}: {e}"
                )

        # Clear the queue
        self._pending_arrangement_analysis.clear()
        print("🎵 All pending arrangement analyses complete!")

    def update_track_arrangement(self, track_id: int, file_path: str):
        """Update a specific track with arrangement analysis"""
        session = self.db.get_session()
        try:
            # Get arrangement analysis
            arrangement = self._classify_track_arrangement(file_path)

            # Update track
            track = session.query(Track).filter(Track.id == track_id).first()
            if track:
                track.raw_arrangement_pattern = arrangement["raw_pattern"]
                track.smoothed_arrangement_pattern = arrangement["smoothed_pattern"]
                track.raw_predictions = arrangement["raw_predictions"]
                track.raw_confidence_scores = arrangement["raw_confidence_scores"]
                session.commit()
            else:
                raise ValueError(f"Track {track_id} not found")

        except Exception as e:
            session.rollback()
            raise
        finally:
            session.close()

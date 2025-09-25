#!/usr/bin/env python3
"""
Backup training examples and feedback to JSON files.

This script exports all training examples, tracks, and feedback to JSON files
that can be restored later if the database needs to be reset.
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import init_app
from db.db import AudioRAGDatabase
from db.models import TrainingExample, Track, Feedback

# Initialize logging and environment
init_app()

import logging

logger = logging.getLogger(__name__)


def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    import numpy as np

    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


def backup_training_data(db_connection_url: str, backup_dir: Path = None):
    """Backup all training data to JSON files."""

    if backup_dir is None:
        backup_dir = Path("data/backups")

    # Create backup directory
    backup_dir.mkdir(parents=True, exist_ok=True)

    # Create timestamped backup folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_folder = backup_dir / f"training_backup_{timestamp}"
    backup_folder.mkdir(exist_ok=True)

    db = AudioRAGDatabase(db_connection_url)
    session = db.get_session()

    try:
        # Backup training examples
        training_examples = session.query(TrainingExample).all()
        training_data = []

        print(f"📦 Backing up {len(training_examples)} training examples...")

        for example in training_examples:
            # Get associated tracks
            example_track = (
                session.query(Track).filter_by(id=example.example_track_id).first()
            )
            reference_track = (
                session.query(Track).filter_by(id=example.reference_track_id).first()
            )

            # Get feedback items
            feedback_items = (
                session.query(Feedback).filter_by(training_example_id=example.id).all()
            )

            example_data = {
                "training_example": {
                    "id": example.id,
                    "created_at": (
                        example.created_at.isoformat() if example.created_at else None
                    ),
                    "genre": example.genre,
                },
                "example_track": (
                    {
                        "id": example_track.id,
                        "file_path": example_track.file_path,
                        "duration": float(example_track.duration),
                        "sample_rate": int(example_track.sample_rate),
                        "global_embedding": (
                            convert_numpy_types(example_track.global_embedding)
                            if example_track.global_embedding is not None
                            else None
                        ),
                        "global_feature_data": (
                            convert_numpy_types(example_track.global_feature_data)
                            if example_track.global_feature_data is not None
                            else None
                        ),
                        "raw_arrangement_pattern": example_track.raw_arrangement_pattern,
                        "smoothed_arrangement_pattern": example_track.smoothed_arrangement_pattern,
                        "raw_predictions": example_track.raw_predictions,
                        "raw_confidence_scores": example_track.raw_confidence_scores,
                        "waveform_viz_path": example_track.waveform_viz_path,
                        "viz_generated_at": (
                            example_track.viz_generated_at.isoformat()
                            if example_track.viz_generated_at
                            else None
                        ),
                    }
                    if example_track
                    else None
                ),
                "reference_track": (
                    {
                        "id": reference_track.id,
                        "file_path": reference_track.file_path,
                        "duration": float(reference_track.duration),
                        "sample_rate": int(reference_track.sample_rate),
                        "global_embedding": (
                            convert_numpy_types(reference_track.global_embedding)
                            if reference_track.global_embedding is not None
                            else None
                        ),
                        "global_feature_data": (
                            convert_numpy_types(reference_track.global_feature_data)
                            if reference_track.global_feature_data is not None
                            else None
                        ),
                        "raw_arrangement_pattern": reference_track.raw_arrangement_pattern,
                        "smoothed_arrangement_pattern": reference_track.smoothed_arrangement_pattern,
                        "raw_predictions": reference_track.raw_predictions,
                        "raw_confidence_scores": reference_track.raw_confidence_scores,
                        "waveform_viz_path": reference_track.waveform_viz_path,
                        "viz_generated_at": (
                            reference_track.viz_generated_at.isoformat()
                            if reference_track.viz_generated_at
                            else None
                        ),
                    }
                    if reference_track
                    else None
                ),
                "feedback_items": [
                    {
                        "id": feedback.id,
                        "feedback_type": feedback.feedback_type,
                        "feedback_text": feedback.feedback_text,
                        "created_at": (
                            feedback.created_at.isoformat()
                            if feedback.created_at
                            else None
                        ),
                    }
                    for feedback in feedback_items
                ],
            }

            training_data.append(example_data)

        # Save training data
        training_file = backup_folder / "training_examples.json"
        with open(training_file, "w") as f:
            json.dump(training_data, f, indent=2, default=str)

        print(f"✅ Training examples saved to: {training_file}")

        # Create backup metadata
        metadata = {
            "backup_timestamp": timestamp,
            "backup_date": datetime.now().isoformat(),
            "training_examples_count": len(training_examples),
            "total_tracks": len(training_examples) * 2,  # Each example has 2 tracks
            "total_feedback_items": sum(
                len(ex["feedback_items"]) for ex in training_data
            ),
            "db_connection_url": (
                db_connection_url.replace(os.getenv("DB_PASSWORD", ""), "***")
                if "DB_PASSWORD" in str(db_connection_url)
                else db_connection_url
            ),
        }

        metadata_file = backup_folder / "backup_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"✅ Backup metadata saved to: {metadata_file}")

        # Create a latest backup symlink for easy access
        latest_link = backup_dir / "latest"
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        latest_link.symlink_to(backup_folder.name)

        print(f"\n📦 Backup completed successfully!")
        print(f"   Backup location: {backup_folder}")
        print(f"   Training examples: {metadata['training_examples_count']}")
        print(f"   Total feedback items: {metadata['total_feedback_items']}")
        print(f"   Latest backup link: {latest_link}")

        return backup_folder

    except Exception as e:
        logger.error(f"Backup failed: {e}")
        print(f"❌ Backup failed: {e}")
        raise
    finally:
        session.close()


def main():
    """Main entry point."""
    from dotenv import load_dotenv

    # Load environment variables
    load_dotenv()

    db_url = os.getenv("DB_CONNECTION_URL")
    if not db_url:
        print("❌ DB_CONNECTION_URL not found in environment")
        return 1

    print("📦 Starting training data backup...")

    try:
        backup_folder = backup_training_data(db_url)
        print(f"\n✅ Backup completed: {backup_folder}")
        return 0
    except Exception as e:
        print(f"❌ Backup failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())

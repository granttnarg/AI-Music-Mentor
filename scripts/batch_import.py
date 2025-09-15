#!/usr/bin/env python3
"""
Batch import script for TrainingExample data.

Scans data/batch_import/ for folders containing input.mp3 and reference.mp3 pairs,
processes audio features, and creates TrainingExample entries with placeholder feedback.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import init_app
from src.audio_features import AudioFeatureService
from db.db import AudioRAGDatabase
from db.operations import AudioRAGOperations

# Initialize logging and environment
init_app()

import logging

logger = logging.getLogger(__name__)


class BatchImporter:
    def __init__(self, db_connection_url: str):
        """Initialize the batch importer with database connection."""
        self.db = AudioRAGDatabase(db_connection_url)
        self.operations = AudioRAGOperations(self.db)
        self.audio_service = AudioFeatureService()
        self.batch_import_dir = Path("data/batch_import")

    def get_default_genre(self) -> str:
        """Return default genre for all imports."""
        return "techno"

    def process_audio_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Process a single audio file and extract features."""
        try:
            logger.info(f"Processing audio file: {file_path}")

            # Load and extract features
            global_features = self.audio_service.load_audio_file(
                file_path
            ).extract_global_features(max_duration=150)

            # Create embedding
            embedding = self.audio_service.create_embedding_vector(global_features)

            # Build feature data
            feature_data = self.audio_service.build_feature_data_object(
                global_features, ["rhythm", "energy"]
            )

            return {
                "file_path": str(file_path),
                "duration": feature_data["metadata"]["duration"],
                "sample_rate": feature_data["metadata"]["sample_rate"],
                "embedding": embedding,
                "success": True,
            }

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return None

    def create_placeholder_feedback(self, folder_name: str) -> List[Dict[str, str]]:
        """Create placeholder feedback items for manual editing later."""
        feedback_items = [
            {
                "feedback_type": "general",
                "feedback_text": f"[EDIT ME] Add your feedback for {folder_name} - what improvements does this track need to reach the reference quality?",
            }
        ]
        return feedback_items

    def find_files_by_prefix(
        self, folder: Path
    ) -> tuple[Optional[Path], Optional[Path]]:
        """Find input and reference files by prefix in a folder."""
        input_file = None
        reference_file = None

        # Support multiple audio formats
        audio_extensions = ["*.mp3", "*.wav", "*.aif"]

        for pattern in audio_extensions:
            for file in folder.glob(pattern):
                if file.name.startswith("input--"):
                    input_file = file
                elif file.name.startswith("ref--"):
                    reference_file = file

        return input_file, reference_file

    def find_track_pairs(self) -> List[Path]:
        """Find all valid track pair folders in batch_import directory."""
        valid_folders = []

        if not self.batch_import_dir.exists():
            logger.error(f"Batch import directory not found: {self.batch_import_dir}")
            return valid_folders

        for folder in self.batch_import_dir.iterdir():
            if not folder.is_dir():
                continue

            input_file, reference_file = self.find_files_by_prefix(folder)

            if input_file and reference_file:
                valid_folders.append(folder)
                logger.info(f"Found valid track pair: {folder.name}")
                logger.info(f"  Input: {input_file.name}")
                logger.info(f"  Reference: {reference_file.name}")
            else:
                missing = []
                if not input_file:
                    missing.append("input--*.mp3")
                if not reference_file:
                    missing.append("ref--*.mp3")
                logger.warning(f"Skipping {folder.name}: missing {', '.join(missing)}")

        return valid_folders

    def import_track_pair(self, folder: Path) -> Optional[int]:
        """Import a single track pair folder."""
        folder_name = folder.name
        logger.info(f"Importing track pair: {folder_name}")

        # Use default genre
        genre = self.get_default_genre()
        logger.info(f"Using default genre: {genre}")

        # Find audio files by prefix
        input_file, reference_file = self.find_files_by_prefix(folder)

        if not input_file or not reference_file:
            logger.error(f"Could not find input-- or ref-- files in {folder_name}")
            return None

        input_data = self.process_audio_file(input_file)
        reference_data = self.process_audio_file(reference_file)

        if not input_data or not reference_data:
            logger.error(f"Failed to process audio files for {folder_name}")
            return None

        # Create placeholder feedback
        feedback_items = self.create_placeholder_feedback(folder_name)

        # Save to database
        try:
            training_id = self.operations.add_training_example(
                input_track_path=input_data["file_path"],
                ref_track_path=reference_data["file_path"],
                input_duration=input_data["duration"],
                input_sample_rate=input_data["sample_rate"],
                input_embedding=input_data["embedding"],
                ref_duration=reference_data["duration"],
                ref_sample_rate=reference_data["sample_rate"],
                ref_embedding=reference_data["embedding"],
                feedback_items=feedback_items,
                genre=genre,
            )

            logger.info(
                f"✅ Created TrainingExample ID: {training_id} for {folder_name}"
            )
            return training_id

        except Exception as e:
            logger.error(f"Database error for {folder_name}: {e}")
            return None

    def run_batch_import(self) -> Dict[str, Any]:
        """Run the complete batch import process."""
        logger.info("Starting batch import process...")

        # Find all valid track pairs
        track_pairs = self.find_track_pairs()

        if not track_pairs:
            logger.warning("No valid track pairs found!")
            return {"success": False, "message": "No valid track pairs found"}

        logger.info(f"Found {len(track_pairs)} track pairs to import")

        # Import each track pair
        successful_imports = []
        failed_imports = []

        for folder in track_pairs:
            training_id = self.import_track_pair(folder)
            if training_id:
                successful_imports.append(
                    {"folder": folder.name, "training_id": training_id}
                )
            else:
                failed_imports.append(folder.name)

        # Summary
        summary = {
            "success": True,
            "total_found": len(track_pairs),
            "successful_imports": len(successful_imports),
            "failed_imports": len(failed_imports),
            "imported_examples": successful_imports,
            "failed_folders": failed_imports,
        }

        logger.info(
            f"Batch import complete: {len(successful_imports)}/{len(track_pairs)} successful"
        )
        return summary


def main():
    """Main entry point for batch import."""
    # Get database connection URL
    connection_url = os.getenv(
        "DB_CONNECTION_URL",
        "postgresql://postgres:your_password@localhost:5434/audio_rag",
    )

    if "your_password" in connection_url:
        logger.error("Please set DB_CONNECTION_URL in your .env file")
        return 1

    # Run batch import
    try:
        importer = BatchImporter(connection_url)
        results = importer.run_batch_import()

        if results["success"]:
            print(f"\n🎵 Batch Import Complete!")
            print(f"✅ Successfully imported: {results['successful_imports']}")
            if results["failed_imports"]:
                print(f"❌ Failed imports: {results['failed_imports']}")
            print(
                f"\n💡 Next step: Use the admin interface to edit placeholder feedback"
            )
            return 0
        else:
            print(f"❌ Batch import failed: {results.get('message', 'Unknown error')}")
            return 1

    except Exception as e:
        logger.error(f"Batch import failed with exception: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

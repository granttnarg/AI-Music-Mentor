from pathlib import Path
from datetime import datetime
from src.audio_features import AudioFeatureService


class UserUploadService:
    """Service for handling user track uploads and processing"""

    def __init__(self):
        self.feature_service = AudioFeatureService()

    def process_and_save_file(
        self, file_content: bytes, filename: str, file_type: str, session_dir: Path
    ) -> dict:
        """
        Process and save a single audio file.

        Args:
            file_content: Raw file bytes
            filename: Original filename
            file_type: "input" or "reference"
            session_dir: Directory to save the file

        Returns:
            Dictionary with processed audio data
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        clean_name = Path(filename).stem
        new_file_info = f"{file_type}--{clean_name}--{timestamp}"
        file_path = session_dir / f"{new_file_info}.mp3"

        with open(file_path, "wb") as f:
            f.write(file_content)

        try:
            global_features = self.feature_service.load_audio_file(
                file_path
            ).extract_global_features(max_duration=400)
            embedding = self.feature_service.create_embedding_vector(global_features)
            feature_data = self.feature_service.build_feature_data_object(
                global_features, ["rhythm", "energy"]
            )

            return {
                "file_path": str(file_path),
                "original_filename": filename,
                "file_size_bytes": len(file_content),
                "duration": feature_data["metadata"]["duration"],
                "sample_rate": feature_data["metadata"]["sample_rate"],
                "embedding": (
                    embedding.tolist() if hasattr(embedding, "tolist") else embedding
                ),
                "global_features": global_features,
                "success": True,
            }
        except Exception as e:
            return {"success": False, "error": str(e), "filename": filename}

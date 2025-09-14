# Batch Import Structure

Place track pairs in individual folders under this directory.

## Folder Structure
```
batch_import/
├── track_name_01/
│   ├── input--my_unfinished_track.mp3      # Unfinished track
│   └── ref--target_finished_track.mp3     # Finished/target track
├── track_name_02/
│   ├── input--another_sketch.mp3
│   └── ref--professional_reference.mp3
```

## Naming Convention
- **Folder names** can be simple like "training_example_01", "training_example_02", etc.
- **File names** must use prefixes:
  - `input--` for unfinished tracks (e.g., `input--my_song_sketch.wav`)
  - `ref--` for reference/target tracks (e.g., `ref--professional_example.mp3`)
  - Supports: MP3, WAV, FLAC, M4A, OGG formats
- You can keep your original filenames after the prefix
- Genre defaults to "deep techno" and can be edited later in the admin interface

## Usage
Run the batch import script from project root:
```bash
uv run python scripts/batch_import.py
```

This will process all folders and create TrainingExample entries with blank feedback that you can edit in the admin interface.
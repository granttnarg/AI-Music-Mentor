from sklearn.externals.array_api_compat.numpy import True_
from db.operations import AudioRAGOperations, AudioRAGDatabase
from db.models import Track
from sqlalchemy import text
import glob
import json

if __name__ == "__main__":

    ## TESTING OUT DB SETUP FOR UPLOADS AND BASIC SIMILARITY SEARCH
    db = AudioRAGDatabase("postgresql://postgres:password@127.0.0.1:5434/audio_rag")
    ops = AudioRAGOperations(db)
    # db.reset_database()

    # Set up schema
    db.setup_database()
    print("setup db...")

    # Find all .mp3.json files
    json_files = glob.glob("uploads/*.mp3.json")
    print(f"Found {len(json_files)} JSON files")

    for json_file in json_files:
        try:
            # Load the JSON data
            with open(json_file, "r", encoding="utf-8") as f:
                audio_data = json.load(f)

                # Check if track already exists
                filename = audio_data["original_filename"]
                if ops.get_track_by_file_path(filename):
                    print(f"⏭️  Skipping {filename} - already exists")
                    continue

                upload_id = ops.add_user_upload(
                    input_track_path=filename,
                    input_duration=audio_data["processed"]["global_feature_data"][
                        "metadata"
                    ]["duration"],
                    input_sample_rate=audio_data["processed"]["global_feature_data"][
                        "metadata"
                    ]["sample_rate"],
                    input_embedding=audio_data["processed"]["global_feature_embedding"],
                    user_prompt="help me finish my arrangment",
                    stage=audio_data["stage"],
                    genre="techno",
                )
                print(f"✅ Uploaded: {filename} (upload_id: {upload_id})")

        except Exception as e:
            print(f"✗ Failed to process {json_file}: {e}")

    ids = db.get_session().query(Track.id, Track.file_path).all()
    print(ids)
    track = ops.get_track(3)  # or whatever ID you want
    print(track)

    if track:
        # Use the embedding from the dictionary
        similar_tracks = ops.find_similar_tracks(
            track["embedding"], limit=5, threshold=0.03
        )
        print(f"Found {len(similar_tracks)} similar tracks")

        for similar in similar_tracks:
            print(f"Similar track: {similar.file_path}")

    session = ops.db.get_session()
    result = session.execute(
        text(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
        )
    )
    tables = result.fetchall()
    print("Tables:", tables)
    session.close()

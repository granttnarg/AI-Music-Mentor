from sklearn.externals.array_api_compat.numpy import True_
from db.operations import AudioRAGOperations, AudioRAGDatabase
from db.models import Track
from sqlalchemy import text
import glob
import json
from collections import defaultdict
import os

if __name__ == "__main__":

    from dotenv import load_dotenv

    load_dotenv()

    # IMPORTING TRACKS FROM OUR JSON SIDECAR SETUP
    ## TESTING OUT DB SETUP FOR UPLOADS AND BASIC SIMILARITY SEARCH
    # TODO: remove json sidecar setup and save from app.py directly into the DB
    connection_url = os.getenv(
        "DB_CONNECTION_URL", "postgresql://postgres:<ADD_TOENV_FILE>"
    )
    db = AudioRAGDatabase(connection_url)
    ops = AudioRAGOperations(db)
    db.reset_database()

    # Set up schema
    db.setup_database()
    print("setup db...")

    # Find all sessions
    session_folders = glob.glob("uploads/session_*")

    # Group files by session_id
    sessions = defaultdict(lambda: {"input": None, "reference": None})

    for session_folder in session_folders:
        json_files = glob.glob(f"{session_folder}/*.mp3.json")
        for json_file in json_files:
            try:
                # Load the JSON data
                with open(json_file, "r", encoding="utf-8") as f:
                    audio_data = json.load(f)

                    session_id = audio_data.get("session_id")
                    file_type = audio_data.get("file_type")  # "input" or "reference"
                    if session_id and file_type:
                        sessions[session_id][file_type] = {
                            "json_file": json_file,
                            "data": audio_data,
                        }
                    else:
                        print(f"ERROR:  Missing session_id or file_type in {json_file}")

            except Exception as e:
                print(f"✗ Failed to read {json_file}: {e}")

    # Process each complete session
    for session_id, files in sessions.items():
        input_file = files["input"]
        ref_file = files["reference"]

        if not input_file or not ref_file:
            print(
                f"⚠️  Incomplete session {session_id} - missing input or reference file"
            )
            continue

        try:
            input_data = input_file["data"]
            ref_data = ref_file["data"]

            # Check if this session already exists (check by session_id or unique combination)
            input_filename = input_data["original_filename"]
            ref_filename = ref_data["original_filename"]

            # You might want to check by session_id instead of filename
            existing_track = ops.get_track_by_file_path(input_filename)
            if existing_track:
                print(f"  Skipping session {session_id} - already exists")
                continue

            upload_id = ops.add_user_upload(
                input_track_path=input_data["original_filename"],
                ref_track_path=ref_data["original_filename"],
                input_duration=input_data["processed"]["global_feature_data"][
                    "metadata"
                ]["duration"],
                input_sample_rate=input_data["processed"]["global_feature_data"][
                    "metadata"
                ]["sample_rate"],
                input_embedding=input_data["processed"]["global_feature_embedding"],
                user_prompt=input_data["user_question"],  # Use the actual user question
                stage=input_data["stage"],
                genre="techno",  # You might want to extract this or make it dynamic
            )
            print(
                f"Uploaded session {session_id}: {input_filename} + {ref_filename} (upload_id: {upload_id})"
            )

        except Exception as e:
            print(f"✗ Failed to process session {session_id}: {e}")
            print(f"   Input file: {input_file['json_file'] if input_file else 'None'}")
            print(f"   Ref file: {ref_file['json_file'] if ref_file else 'None'}")

    ids = db.get_session().query(Track.id, Track.file_path).all()
    print(f"\nTotal tracks in DB: {len(ids)}")
    for track_id, file_path in ids:
        print(f"  Track {track_id}: {file_path}")

    # Test similarity search
    if ids:
        track = ops.get_track(ids[0][0])  # Get the first track
        print(
            f"\nTesting similarity search with track: {track['file_path'] if track else 'None'}"
        )

        if track:
            # Use the embedding from the dictionary
            similar_tracks = ops.find_similar_tracks(
                track["embedding"], limit=5, threshold=0.5
            )
            print(f"Found {len(similar_tracks)} similar tracks")

            for similar in similar_tracks:
                print(f"  Similar track: {similar.file_path}")

    # Check database tables
    session = ops.db.get_session()
    result = session.execute(
        text(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
        )
    )
    tables = result.fetchall()
    print(f"\nDatabase tables: {[table[0] for table in tables]}")
    session.close()

import os
from src.audio_features import AudioFeatureService

def main():
    file_path = './data/test/2025-07-21 Nik Fury Remix.mp3'

    service = AudioFeatureService()
    global_features = service.load_audio_file(file_path).extract_global_features()
    print(global_features)

    raw_vector = service.create_embedding_vector(global_features)
    print(raw_vector)

if __name__ == "__main__":
    main()


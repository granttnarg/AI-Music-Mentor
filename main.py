import os
from src.audio_features import AudioFeatureService
from pprint import pprint

def main():
    file_path = './data/test/2025-07-21 Nik Fury Remix.mp3'

    #init service for feature extraction
    service = AudioFeatureService()
    global_features = service.load_audio_file(file_path).extract_global_features()
    pprint(global_features)
    print('###')

    #embed all features into one single vector for similarity search
    raw_vector = service.create_embedding_vector(global_features)
    print(raw_vector)
    print('###')

    #extract feature into an object we can then append to a db entry
    feature_data_for_db = service.build_feature_data_object(global_features,['rhythm', 'energy'])
    pprint(feature_data_for_db)


if __name__ == "__main__":
    main()



import numpy as np
from feature_extraction import AudioProcess


def predict_song(audio_path, model, class_names):
    """
    Predict arrangement sections for a single song using new_model.

    Parameters:
    - audio_path: str, path to the audio file
    - model: tf.keras.Model, trained model (e.g. new_model)
    - class_names: list of str, class labels used in training

    Returns:
    - predicted_classes: NumPy array of predicted class indices per meter
    - meter_grid: NumPy array of meter grid frames
    """
    # Preprocess the song (same as training pipeline)
    audio_processor = AudioProcess()
    padded_song, audio_features = audio_processor.process_audio(audio_path)
    if padded_song is None:
        print("❌ Failed to process song")
        return None, None

    # Run the model
    preds = model.predict(padded_song).squeeze(axis=0)  # shape: (meters, classes)

    # Convert to class indices
    predicted_classes = np.argmax(preds, axis=-1)

    return predicted_classes, audio_features.meter_grid


# _, audio_features = process_audio(full_path_wip)
# visualize_predictions_complete(audio_features, pred_labels, class_names)
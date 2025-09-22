import tensorflow as tf
import json
import os


def load_energy_classifier(
    model_dir="/content/drive/MyDrive/DSR-AI-MENTOR/final_classification_models/energy_classifier_4class_5sec_20250920_1257",
):
    """Load the saved energy classifier model"""

    # Load model
    model_path = os.path.join(model_dir, "model.h5")
    model = tf.keras.models.load_model(model_path)

    # Load config
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    # Load class weights
    weights_path = os.path.join(model_dir, "class_weights.json")
    with open(weights_path, "r") as f:
        class_weights = json.load(f)

    print(f"Loaded model: {config['model_name']}")
    print(f"Classes: {config['class_names']}")
    print(f"Input shape: {config['input_shape']}")

    return model, config, class_weights


# Usage:
# model, config, class_weights = load_energy_classifier()

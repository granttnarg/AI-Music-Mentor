"""
Arrangement Classification Model for Audio Segmentation.

This module provides functionality to load and use the arrangement classifier model
for analyzing audio segments and returning timestamps for further processing.
"""

import os
import json
import logging
import numpy as np
import tensorflow as tf
from typing import List, Tuple, Dict, Optional
from pathlib import Path

## HELPER METHODS FOR OUR CRNN MODEL

# Model Arch Pre FineTunning
def create_original_crnn_model(max_frames_per_meter=300, max_meters=201, n_features=15):
    """Recreate the exact CRNN model architecture."""
    frame_input = tf.keras.layers.Input(shape=(max_frames_per_meter, n_features))
    conv1 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(frame_input)
    pool1 = tf.keras.layers.MaxPooling1D(pool_size=2, padding='same')(conv1)
    conv2 = tf.keras.layers.Conv1D(filters=256, kernel_size=3, activation='relu', padding='same')(pool1)
    pool2 = tf.keras.layers.MaxPooling1D(pool_size=2, padding='same')(conv2)
    conv3 = tf.keras.layers.Conv1D(filters=256, kernel_size=3, activation='relu', padding='same')(pool2)
    pool3 = tf.keras.layers.MaxPooling1D(pool_size=2, padding='same')(conv3)
    frame_features = tf.keras.layers.Flatten()(pool3)
    frame_feature_model = tf.keras.Model(inputs=frame_input, outputs=frame_features)

    meter_input = tf.keras.layers.Input(shape=(max_meters, max_frames_per_meter, n_features))
    time_distributed = tf.keras.layers.TimeDistributed(frame_feature_model)(meter_input)
    masking_layer = tf.keras.layers.Masking(mask_value=0.0)(time_distributed)
    lstm_out = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True))(masking_layer)
    output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1, activation='sigmoid'))(lstm_out)

    model = tf.keras.Model(inputs=meter_input, outputs=output)
    return model

def masked_categorical_crossentropy(y_true, y_pred):
    """Custom loss function that handles padded sequences."""
    # Create a mask from the true labels (assuming padding is represented by all zeros or -1s)
    mask = tf.reduce_sum(tf.cast(tf.not_equal(y_true, -1.0), tf.float32), axis=-1)
    mask = tf.cast(tf.not_equal(mask, 0), tf.float32) # Mask is 1 where there is data, 0 where padded

    # Calculate categorical crossentropy
    cce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)

    # Apply mask
    masked_cce = cce * mask

    # Return average loss (only over non-padded elements)
    return tf.reduce_sum(masked_cce) / (tf.reduce_sum(mask) + tf.keras.backend.epsilon()) # Add epsilon for numerical stability

def masked_accuracy(y_true, y_pred):
    """Custom accuracy metric that handles padded sequences."""
    # Create a mask from the true labels (assuming padding is represented by all zeros or -1s)
    mask = tf.reduce_sum(tf.cast(tf.not_equal(y_true, -1.0), tf.float32), axis=-1)
    mask = tf.cast(tf.not_equal(mask, 0), tf.float32) # Mask is 1 where there is data, 0 where padded

    # Get predictions and true labels (ignoring padding)
    y_pred_classes = tf.argmax(y_pred, axis=-1)
    y_true_classes = tf.argmax(y_true, axis=-1)

    # Apply mask to true and predicted classes
    y_true_masked = y_true_classes * tf.cast(mask, tf.int64)
    y_pred_masked = y_pred_classes * tf.cast(mask, tf.int64)

    # Calculate accuracy only on non-padded elements
    correct = tf.cast(tf.equal(y_pred_masked, y_true_masked), tf.float32) * mask

    return tf.reduce_sum(correct) / (tf.reduce_sum(mask) + tf.keras.backend.epsilon()) # Add epsilon for numerical stability

# Model Arch Post FineTunning - 4 Class output [ O, A, B, C ]
def modify_model_for_multiclass_recreated(model, num_classes=4):
    """Recreate the exact multiclass modification with proper loss functions."""
    for layer in model.layers:
        layer.trainable = False

    # Using the most succesful head from our finetuning experiments, freezing the rest of the model.
    x = model.layers[-2].output
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(256, activation='relu'))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(0.3))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(128, activation='relu'))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(0.3))(x)
    new_output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(num_classes, activation='softmax'))(x)

    new_model = tf.keras.Model(inputs=model.input, outputs=new_output)

    # Compile with custom loss functions, same used in original model.
    new_model.compile(
        optimizer='adam',
        loss=masked_categorical_crossentropy,
        metrics=[masked_accuracy]
    )

    return new_model


class ArrangementClassifier:
    """
    Arrangement section classifier for audio segmentation.

    Classifies 5-second audio segments into arrangement sections:
    - O (Other): Intro/outro/other sections
    - A (Medium Energy): Medium energy content
    - B (High Energy): High energy sections like drops, climaxes
    - C (Breakdown): Breakdown/transition sections
    """
    def __init__(self, model_dir: Optional[str] = None):
        """
        Initialize the arrangement classifier.

        Args:
            model_dir: Directory containing the saved model files. If None, will look for default location.
        """
        self.model = None
        self.config = None
        self.class_weights = None
        self.class_names = ['O', 'A', 'B', 'C']  # Default class names
        self.model_dir = model_dir
        self.segment_length_seconds = 5 # Default to 5 seconds as it was more reliable than 1.7 seconds used in original model.
        self.sample_rate = 12000
        self.hop_length = 128
        self.n_features = 15

    def load_model(self, model_dir: Optional[str] = 'models/arrangement_classifier/4classes') -> bool:
        """
        Load the arrangement classifier model and configuration using the tested CRNN approach.

        Args:
            model_dir: Directory containing the saved model files

        Returns:
            bool: True if model loaded successfully, False otherwise
        """

        self.model_dir = model_dir

        if not self.model_dir or not os.path.exists(self.model_dir):
            print(f"Error: Model directory not found. Please specify the correct path.")
            return False

        try:
            # Load configuration first
            config_path = os.path.join(self.model_dir, "config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    self.config = json.load(f)

                # Update parameters from config
                self.class_names = self.config.get('class_names', self.class_names)
                self.segment_length_seconds = self.config.get('segment_length_seconds', 5)
                self.sample_rate = self.config.get('sample_rate', 12000)
                self.hop_length = self.config.get('hop_length', 128)
                self.n_features = self.config.get('n_features', 15)

                print(f"Configuration loaded. Classes: {self.class_names}")
            else:
                print("ERROR: Configuration file not found, using default parameters")

            # Create and load model using tested approach
            model_path = os.path.join(self.model_dir, "model.h5")

            if not os.path.exists(model_path):
                print(f"Error: Model file not found at {model_path}")
                return False

            # Recreate the exact model architecture and load weights
            original_model = create_original_crnn_model()
            multiclass_model = modify_model_for_multiclass_recreated(original_model, num_classes=4)

            # Load weights with custom objects if needed
            try:
                multiclass_model.load_weights(model_path)
            except Exception as e:
                print(f"Error loading weights directly: {e}")
                print("Attempting to load with custom objects...")
                # If direct loading fails, try loading the full model with custom objects
                custom_objects = {
                    'masked_categorical_crossentropy': masked_categorical_crossentropy,
                    'masked_accuracy': masked_accuracy
                }
                multiclass_model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
                print("Model loaded with custom objects")

            self.model = multiclass_model
            print(f"Arrangement classifier model loaded from {model_path}")

            # Load class weights
            weights_path = os.path.join(self.model_dir, "class_weights.json")
            if os.path.exists(weights_path):
                with open(weights_path, 'r') as f:
                    self.class_weights = json.load(f)
                print(f"Class weights loaded")
            else:
                print("⚠️  Class weights file not found")

            return True

        except Exception as e:
            print(f"❌ Error loading arrangement classifier model: {e}")
            return False

    def process_audio_with_crnn(self, audio_path: str, trim_silence: bool = True) -> Tuple[np.ndarray, np.ndarray, any]:
        """
        Process audio using the tested CRNN pipeline (meter-based approach).

        Args:
            audio_path: Path to the audio file
            trim_silence: Whether to trim silence from audio

        Returns:
            Tuple of (predictions, confidence_scores, audio_features)
        """
        if not self.model:
            if not self.load_model():
                print("❌ Failed to load model")
                return None, None, None

        print(f"Analyzing arrangement with CRNN pipeline: {Path(audio_path).name}")

        try:
            # Import AudioProcess here to avoid circular imports
            from src.classifier.feature_extraction import AudioProcess

            # Process audio with CRNN pipeline
            audio_processor = AudioProcess()
            processed_audio, audio_features = audio_processor.process_audio(
                audio_path, trim_silence=trim_silence, sr=self.sample_rate, hop_length=self.hop_length
            )

            if processed_audio is None:
                print("❌ Audio processing failed")
                return None, None, None

            print(f"Audio processed succesfully: {processed_audio.shape}")

            # Get predictions
            print("Running inference...")
            predictions = self.model.predict(processed_audio, verbose=0)
            predicted_classes = np.argmax(predictions[0], axis=1)
            confidence_scores = np.max(predictions[0], axis=1)

            # Limit to actual audio length
            actual_meters = min(len(audio_features.meter_grid) - 1, len(predicted_classes))
            predicted_classes = predicted_classes[:actual_meters]
            confidence_scores = confidence_scores[:actual_meters]

            print(f"Predictions complete: {predictions.shape}")
            print(f"   Analyzing {len(predicted_classes)} segments")

            # Show quick stats
            unique, counts = np.unique(predicted_classes, return_counts=True)
            print(f"\n📊 Raw predictions stats:")
            for class_idx, count in zip(unique, counts):
                class_name = self.class_names[class_idx]
                percentage = (count / len(predicted_classes)) * 100
                print(f"   {class_name}: {count} segments ({percentage:.1f}%)")

            # Log predictions for debugging
            predictions_logger = logging.getLogger('predictions')
            predictions_logger.info(f"PREDICTION | {Path(audio_path).name} | {len(predicted_classes)} segments | "
                                  f"Pattern: {'-'.join([self.class_names[c] for c in predicted_classes])} | "
                                  f"Avg confidence: {confidence_scores.mean():.3f}")

            return predicted_classes, confidence_scores, audio_features

        except Exception as e:
            print(f"❌ Error processing audio with CRNN: {e}")
            return None, None, None

    def get_smoothed_blocks(self, raw_predictions: List[int], confidence_scores: List[float], 
                           min_segment_length: int = 2, confidence_threshold: float = 0.4) -> Tuple[List[Dict], Dict]:
        """
        Helper method to generate smoothed blocks from raw predictions.
        
        Args:
            raw_predictions: List of raw prediction class indices
            confidence_scores: List of confidence scores
            min_segment_length: Minimum segments for a section
            confidence_threshold: Confidence threshold for filtering
            
        Returns:
            Tuple of (blocks, analysis)
        """
        from classifier.arrangement_postprocessing import process_arrangement_predictions
        import numpy as np
        
        predictions_array = np.array(raw_predictions)
        confidence_array = np.array(confidence_scores)
        
        return process_arrangement_predictions(
            predictions_array, confidence_array, self.class_names,
            min_segment_length=min_segment_length,
            confidence_threshold=confidence_threshold
        )

    def _compress_pattern_with_counts(self, predictions: List[int]) -> str:
        """
        Compress raw predictions into a pattern with segment counts.
        
        Args:
            predictions: List of class indices
            
        Returns:
            Compressed pattern string like "1A-7O-4A-16B-1A-1C-2A"
        """
        if not predictions:
            return ""
        
        compressed = []
        i = 0
        
        while i < len(predictions):
            current_class = predictions[i]
            segment_start = i
            
            # Count consecutive segments of same class
            while i < len(predictions) and predictions[i] == current_class:
                i += 1
            
            segment_length = i - segment_start
            class_name = self.class_names[current_class]
            
            compressed.append(f"{segment_length}{class_name}")
        
        return '-'.join(compressed)

    def analyze_arrangement_structure(self, audio_path: str, min_segment_length: int = 2,
                                    confidence_threshold: float = 0.4) -> Dict | None:
        """
        Analyze audio and return both raw and smoothed arrangement data for database storage.

        Args:
            audio_path: Path to the audio file
            min_segment_length: Minimum segments for a section (for postprocessing)
            confidence_threshold: Confidence threshold for filtering (for postprocessing)

        Returns:
            Dict with raw and smoothed arrangement data, or None if failed
            Example: {
                'raw_pattern': 'A-O-O-O-A-A-B-B-...',
                'smoothed_pattern': 'O-A-B-A-C',
                'raw_predictions': [1,0,0,0,1,1,2,2,...],
                'raw_confidence_scores': [0.64,0.95,0.67,...]
            }
        """
        try:
            # Get raw predictions from CRNN
            predicted_classes, confidence_scores, _audio_features = self.process_audio_with_crnn(audio_path)

            if predicted_classes is None:
                return None

            # Create compressed raw pattern with counts
            raw_pattern = self._compress_pattern_with_counts(predicted_classes.tolist())

            # Generate smoothed pattern using helper method
            blocks, analysis = self.get_smoothed_blocks(
                predicted_classes.tolist(), confidence_scores.tolist(),
                min_segment_length=min_segment_length,
                confidence_threshold=confidence_threshold
            )
            smoothed_pattern = '-'.join(analysis['section_sequence'])

            print(f"Arrangement analysis complete:")
            print(f"   Raw pattern: {raw_pattern}")
            print(f"   Smoothed pattern: {smoothed_pattern}")
            print(f"   Smoothed blocks: {len(blocks)} sections")
            print(f"   Structure type: {analysis['structure_type']}")

            return {
                'raw_pattern': raw_pattern,
                'smoothed_pattern': smoothed_pattern,
                'raw_predictions': predicted_classes.tolist(),
                'raw_confidence_scores': confidence_scores.tolist()
            }

        except Exception as e:
            print(f"❌ Error analyzing arrangement structure: {e}")
            return None, None

    def get_class_distribution(self, timestamped_segments: List[Dict]) -> Dict:
        """
        Get distribution statistics for classified segments.

        Args:
            timestamped_segments: List of timestamped segment dictionaries

        Returns:
            Dictionary with distribution statistics
        """
        if not timestamped_segments:
            return {}

        class_counts = {class_name: 0 for class_name in self.class_names}
        class_durations = {class_name: 0.0 for class_name in self.class_names}
        total_duration = 0.0

        for segment in timestamped_segments:
            section = segment['arrangement_section']
            duration = segment['duration']

            class_counts[section] += 1
            class_durations[section] += duration
            total_duration += duration

        # Calculate percentages
        class_percentages = {
            class_name: (class_durations[class_name] / total_duration) * 100 if total_duration > 0 else 0
            for class_name in self.class_names
        }

        return {
            'total_segments': len(timestamped_segments),
            'total_duration': total_duration,
            'class_counts': class_counts,
            'class_durations': class_durations,
            'class_percentages': class_percentages,
            'average_confidence': np.mean([s['confidence'] for s in timestamped_segments])
        }



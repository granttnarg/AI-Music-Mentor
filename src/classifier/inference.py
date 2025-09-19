

import os

folder_path = "models"  # or wherever your model is stored
crnn_model = "best_model_V3.h5"

MODEL_PATH = os.path.join(folder_path, crnn_model)

"""
Model functionality for chorus detection.
"""

import os
import numpy as np
import tensorflow as tf
import librosa

class CRCNNModel:
    def __init__(self):
        pass

    def create_crnn_model(self,max_frames_per_meter=300, max_meters=201, n_features=15):
        """
        Recreate the exact CRNN model architecture from the repo
        """
        # Frame-level feature extractor (CNN part)
        frame_input = tf.keras.layers.Input(shape=(max_frames_per_meter, n_features))
        conv1 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(frame_input)
        pool1 = tf.keras.layers.MaxPooling1D(pool_size=2, padding='same')(conv1)
        conv2 = tf.keras.layers.Conv1D(filters=256, kernel_size=3, activation='relu', padding='same')(pool1)
        pool2 = tf.keras.layers.MaxPooling1D(pool_size=2, padding='same')(conv2)  # Fixed: was pool2, should be conv2
        conv3 = tf.keras.layers.Conv1D(filters=256, kernel_size=3, activation='relu', padding='same')(pool2)
        pool3 = tf.keras.layers.MaxPooling1D(pool_size=2, padding='same')(conv3)
        frame_features = tf.keras.layers.Flatten()(pool3)
        frame_feature_model = tf.keras.Model(inputs=frame_input, outputs=frame_features)

        # Full model with LSTM
        meter_input = tf.keras.layers.Input(shape=(max_meters, max_frames_per_meter, n_features))
        time_distributed = tf.keras.layers.TimeDistributed(frame_feature_model)(meter_input)
        masking_layer = tf.keras.layers.Masking(mask_value=0.0)(time_distributed)
        lstm_out = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True))(masking_layer)
        output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1, activation='sigmoid'))(lstm_out)

        model = tf.keras.Model(inputs=meter_input, outputs=output)
        return model

    def load_CRNN_model(self, model_path: str = MODEL_PATH) -> tf.keras.Model:
        """Load a pre-trained CRNN model from the specified path."""
        try:
            # Create the model architecture
            model = self.create_crnn_model()

            # Load just the weights
            model.load_weights(model_path)
            print("Model loaded successfully!")
            return model

        except Exception as e:
            print(f"Error loading model: {e}")
            return None


    def make_predictions(self, model, processed_audio, audio_features):
        """Make chorus predictions using the loaded model."""
        # Generate predictions
        raw_predictions = model.predict(processed_audio).squeeze()

        # Limit predictions to actual meters
        n_meters = min(len(audio_features.meter_grid) - 1, len(raw_predictions))
        predictions = raw_predictions[:n_meters]

        # Apply smoothing
        smoothed_predictions = self._smooth_predictions(predictions)

        # Calculate time values for display
        meter_grid_times = librosa.frames_to_time(
            audio_features.meter_grid, sr=audio_features.sr, hop_length=audio_features.hop_length)

        # Find chorus segments
        chorus_indices = np.where(smoothed_predictions == 1)[0]
        chorus_start_times = []
        chorus_end_times = []

        if len(chorus_indices) > 0:
            # Group consecutive indices
            groups = []
            current_group = [chorus_indices[0]]

            for i in range(1, len(chorus_indices)):
                if chorus_indices[i] == chorus_indices[i-1] + 1:
                    current_group.append(chorus_indices[i])
                else:
                    groups.append(current_group)
                    current_group = [chorus_indices[i]]
            groups.append(current_group)

            # Display chorus segments
            print("\nDetected chorus sections:")
            for i, group in enumerate(groups):
                start_time = meter_grid_times[group[0]]
                end_time = meter_grid_times[group[-1] + 1]
                chorus_start_times.append(start_time)
                chorus_end_times.append(end_time)

                start_min, start_sec = divmod(start_time, 60)
                end_min, end_sec = divmod(end_time, 60)

                print(f"Chorus {i+1}: {int(start_min)}:{start_sec:05.2f} - {int(end_min)}:{end_sec:05.2f}")
        else:
            print("No choruses detected in this audio file.")

        return smoothed_predictions, chorus_start_times, chorus_end_times

    def modify_model_for_multiclass(self, model, class_names):
        """Modified version that handles custom loss for padding."""
        num_classes = len(class_names)

        # Freeze all layers except the last one
        for layer in model.layers:
            layer.trainable = False

        # Remove the last TimeDistributed Dense layer
        x = model.layers[-2].output

        # Add dropout before the output layer -> didnt help on small dataset. maybe try it again when we have more training data.
        # x = tf.keras.layers.TimeDistributed(
        #     tf.keras.layers.Dropout(0.3)
        # )(x)

        # Add new multiclass output layer
        new_output = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(num_classes, activation='softmax')
        )(x)

        # Create and compile new model
        new_model = tf.keras.Model(inputs=model.input, outputs=new_output)
        new_model.compile(
            optimizer='adam',
            loss=self._masked_categorical_crossentropy,
            metrics=[self._masked_accuracy]
        )

        return new_model

    # PRIVATE METHODS

    def _smooth_predictions(self, data: np.ndarray) -> np.ndarray:
        """Apply smoothing to model predictions to reduce jitter."""
        # First pass: Moving average
        window_size = 3
        smoothed = np.zeros_like(data)
        for i in range(len(data)):
            window_start = max(0, i - window_size // 2)
            window_end = min(len(data), i + window_size // 2 + 1)
            smoothed[i] = np.mean(data[window_start:window_end])

        # Second pass: Eliminate short segments
        min_segment_length = 2
        current_segment_length = 1
        current_value = smoothed[0] > 0.5
        binary_smoothed = np.zeros_like(smoothed, dtype=int)
        binary_smoothed[0] = int(current_value)

        for i in range(1, len(smoothed)):
            new_value = smoothed[i] > 0.5
            if new_value == current_value:
                current_segment_length += 1
            else:
                # If segment is too short, revert to previous value
                if current_segment_length < min_segment_length:
                    for j in range(i - current_segment_length, i):
                        binary_smoothed[j] = int(new_value)
                current_value = new_value
                current_segment_length = 1
            binary_smoothed[i] = int(current_value)

        # Third pass: Fix final segment if too short
        if current_segment_length < min_segment_length:
            for j in range(len(smoothed) - current_segment_length, len(smoothed)):
                binary_smoothed[j] = int(not current_value)

        return binary_smoothed

    def _masked_categorical_crossentropy(self, y_true, y_pred):
        # Create a mask from the true labels (assuming padding is represented by all zeros or -1s)
        mask = tf.reduce_sum(tf.cast(tf.not_equal(y_true, -1.0), tf.float32), axis=-1)
        mask = tf.cast(tf.not_equal(mask, 0), tf.float32) # Mask is 1 where there is data, 0 where padded

        # Calculate categorical crossentropy
        cce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)

        # Apply mask
        masked_cce = cce * mask

        # Return average loss (only over non-padded elements)
        return tf.reduce_sum(masked_cce) / (tf.reduce_sum(mask) + tf.keras.backend.epsilon()) # Add epsilon for numerical stability

    def _masked_accuracy(self, y_true, y_pred):
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
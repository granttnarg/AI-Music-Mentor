"""
Architecture testing for CRNN fine-tuning with different head configurations.
Designed to work in notebooks with your existing classifier code.
"""

import tensorflow as tf
import numpy as np
from inference import CRCNNModel
import wandb
from datetime import datetime
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

class CRNNArchitectureTester:
    def __init__(self, base_model_path, project_name="AI-Mentor_DSR-3&4_Class_Classifier_arch_tests"):
        """
        Initialize architecture tester for CRNN models

        Args:
            base_model_path: Path to your trained CRNN model (best_model_V3.h5)
            project_name: W&B project name
        """
        self.base_model_path = base_model_path
        self.project_name = project_name
        self.crnn_model = CRCNNModel()

    def setup_experiment(self, experiment_config):
        """Initialize W&B run with configuration"""
        wandb.init(
            project=self.project_name,
            config=experiment_config,
            name=f"{experiment_config['architecture_name']}_{experiment_config['num_classes']}class",
            tags=[
                f"{experiment_config['num_classes']}-class",
                experiment_config['architecture_name']
            ]
        )
        return wandb.config

    def create_architecture_variant(self, config, base_model):
        """
        Create different architecture variants based on config.
        Keeps the CNN+LSTM base and only modifies the head.
        """
        # Get the LSTM output (before the final dense layer)
        lstm_output = base_model.layers[-2].output  # BiLSTM output

        # Create new head based on architecture type
        x = lstm_output

        if config.architecture_name == 'option1_dense_layers':
            # Option 1: Add Dense Layers Before Final Output
            x = tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(128, activation='relu', name='dense_128')
            )(x)
            x = tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(64, activation='relu', name='dense_64')
            )(x)
            x = tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(config.num_classes, activation='softmax', name='predictions')
            )(x)

        elif config.architecture_name == 'option2_larger_lstm':
            # Option 2: Rebuild with larger LSTM units
            # Need to rebuild from frame features
            frame_features = base_model.layers[-4].output  # TimeDistributed output
            masking_layer = tf.keras.layers.Masking(mask_value=0.0)(frame_features)

            # Larger LSTM
            lstm_units = config.get('lstm_units', 512)
            x = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(lstm_units, return_sequences=True, name=f'lstm_{lstm_units}')
            )(masking_layer)
            x = tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(config.num_classes, activation='softmax', name='predictions')
            )(x)

        elif config.architecture_name == 'option3_stacked_lstm':
            # Option 3: Add More LSTM Layers
            frame_features = base_model.layers[-4].output
            masking_layer = tf.keras.layers.Masking(mask_value=0.0)(frame_features)

            # First LSTM layer
            x = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(256, return_sequences=True, name='lstm_1')
            )(masking_layer)
            # Second LSTM layer
            x = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(256, return_sequences=True, name='lstm_2')
            )(x)
            x = tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(config.num_classes, activation='softmax', name='predictions')
            )(x)

        elif config.architecture_name == 'option4_combination':
            # Option 4: Combination Approach (Recommended)
            frame_features = base_model.layers[-4].output
            masking_layer = tf.keras.layers.Masking(mask_value=0.0)(frame_features)

            # Larger LSTM
            x = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(512, return_sequences=True, name='lstm_512')
            )(masking_layer)
            # Dense layer
            x = tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(256, activation='relu', name='dense_256')
            )(x)
            # Dropout for regularization
            x = tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dropout(0.3, name='dropout')
            )(x)
            # Final output
            x = tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(config.num_classes, activation='softmax', name='predictions')
            )(x)

        elif config.architecture_name == 'baseline':
            # Baseline: Just change the output classes
            x = tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(config.num_classes, activation='softmax', name='predictions')
            )(x)

        # Create new model
        new_model = tf.keras.Model(inputs=base_model.input, outputs=x, name=config.architecture_name)

        # Compile with your custom loss functions
        new_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
            loss=self.crnn_model._masked_categorical_crossentropy,
            metrics=[self.crnn_model._masked_accuracy]
        )

        return new_model

    def modify_pretrained_model(self, config):
        """
        Load base model and modify for multi-class classification
        """
        try:
            # Load the base CRNN model
            base_model = self.crnn_model.create_crnn_model()
            base_model.load_weights(self.base_model_path)

            # Freeze all layers except the ones we're modifying
            for layer in base_model.layers[:-2]:  # Keep LSTM trainable for some fine-tuning
                layer.trainable = False
                
            # Create architecture variant
            model = self.create_architecture_variant(config, base_model)
            
            # Log model info
            wandb.config.update({
                'total_params': model.count_params(),
                'trainable_params': sum([tf.keras.utils.count_params(w) for w in model.trainable_weights]),
                'model_layers': len(model.layers)
            })
            
            return model
            
        except Exception as e:
            print(f"Error loading/modifying model: {e}")
            return None

    def train_with_wandb(self, model, X_train, y_train, X_val, y_val, config):
        """Train model with comprehensive W&B logging and model saving"""
        
        # Create model checkpoint callback for local saving
        model_save_path = f"models/{config.architecture_name}_{config.num_classes}class_{wandb.run.id}"
        os.makedirs("models", exist_ok=True)
        
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=f"{model_save_path}_best.h5",
            monitor='val_masked_accuracy',
            mode='max',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
        
        # Enhanced W&B callback with model saving
        wandb_callback = wandb.keras.WandbCallback(
            monitor='val_masked_accuracy',
            mode='max',
            save_model=True,  # Save to W&B
            save_graph=True,
            save_weights_only=False,
            log_weights=True,  # Log weight histograms
            log_gradients=True  # Log gradient histograms
        )
        
        # Custom callback for detailed per-epoch logging
        class DetailedLoggingCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                # Log additional metrics to W&B
                wandb.log({
                    'epoch': epoch,
                    'train_loss': logs.get('loss', 0),
                    'train_accuracy': logs.get('masked_accuracy', 0),
                    'val_loss': logs.get('val_loss', 0),
                    'val_accuracy': logs.get('val_masked_accuracy', 0),
                    'learning_rate': float(self.model.optimizer.learning_rate)
                })
        
        callbacks = [
            wandb_callback,
            checkpoint_callback,
            DetailedLoggingCallback(),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_masked_accuracy',
                patience=config.early_stopping_patience,
                restore_best_weights=True,
                mode='max',
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Train with enhanced logging
        print(f"🚀 Training {config.architecture_name} with {config.num_classes} classes...")
        print(f"💾 Models will be saved to: {model_save_path}")
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=config.epochs,
            batch_size=config.batch_size,
            class_weight=getattr(config, 'class_weights', None),
            callbacks=callbacks,
            verbose=1
        )
        
        # Log comprehensive final metrics
        final_metrics = {
            'final_train_acc': history.history['masked_accuracy'][-1],
            'final_val_acc': history.history['val_masked_accuracy'][-1],
            'final_train_loss': history.history['loss'][-1],
            'final_val_loss': history.history['val_loss'][-1],
            'best_val_acc': max(history.history['val_masked_accuracy']),
            'best_val_acc_epoch': np.argmax(history.history['val_masked_accuracy']) + 1,
            'epochs_trained': len(history.history['loss']),
            'improvement_over_baseline': max(history.history['val_masked_accuracy']) - history.history['val_masked_accuracy'][0],
            'model_save_path': model_save_path
        }
        
        # Save final model locally too
        model.save(f"{model_save_path}_final.h5")
        print(f"💾 Final model saved to: {model_save_path}_final.h5")
        
        # Log final metrics
        wandb.log(final_metrics)
        
        # Log model artifacts to W&B
        wandb.save(f"{model_save_path}_best.h5")
        wandb.save(f"{model_save_path}_final.h5")
        
        return history, model_save_path

    def evaluate_detailed_performance(self, model, X_test, y_test, class_names):
        """
        Comprehensive evaluation with precision, recall, F1 logged to W&B
        """
        from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
        
        print("📊 Evaluating detailed performance...")
        
        # Get predictions
        y_pred_proba = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=-1)
        y_true = np.argmax(y_test, axis=-1)
        
        # Handle sequence data (flatten if needed)
        if len(y_pred.shape) > 1:
            y_pred_flat = y_pred.flatten()
            y_true_flat = y_true.flatten()
        else:
            y_pred_flat = y_pred
            y_true_flat = y_true
        
        # Remove masked values (-1 indicates padding)
        mask = y_true_flat >= 0
        y_pred_flat = y_pred_flat[mask]
        y_true_flat = y_true_flat[mask]
        
        print(f"Evaluating {len(y_pred_flat)} valid predictions")
        
        # Calculate detailed metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true_flat, y_pred_flat, average=None, labels=range(len(class_names))
        )
        
        # Calculate macro and weighted averages
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true_flat, y_pred_flat, average='macro'
        )
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true_flat, y_pred_flat, average='weighted'
        )
        
        # Overall accuracy
        accuracy = (y_pred_flat == y_true_flat).mean()
        
        # Create detailed metrics dict
        detailed_metrics = {
            'overall_accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted,
        }
        
        # Per-class metrics
        for i, class_name in enumerate(class_names):
            if i < len(precision):
                detailed_metrics.update({
                    f'{class_name}_precision': float(precision[i]),
                    f'{class_name}_recall': float(recall[i]),
                    f'{class_name}_f1': float(f1[i]),
                    f'{class_name}_support': int(support[i]),
                })
        
        # Log to W&B
        wandb.log(detailed_metrics)
        
        # Create and log confusion matrix
        cm = confusion_matrix(y_true_flat, y_pred_flat)
        wandb.log({
            "confusion_matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=y_true_flat,
                preds=y_pred_flat,
                class_names=class_names
            )
        })
        
        # Print summary
        print("\\n📈 Performance Summary:")
        print(f"Overall Accuracy: {accuracy:.4f}")
        print(f"Macro F1: {f1_macro:.4f}")
        print(f"Weighted F1: {f1_weighted:.4f}")
        
        print("\\nPer-class Performance:")
        for i, class_name in enumerate(class_names):
            if i < len(precision):
                print(f"{class_name:8s}: P={precision[i]:.3f} R={recall[i]:.3f} F1={f1[i]:.3f} Support={support[i]}")
        
        return detailed_metrics

def load_pickle_data(segments_path, labels_path):
    """
    Load pickled segments and labels data
    
    Returns processed X_train, X_val, y_train, y_val ready for training
    """
    print(f"Loading segments from: {segments_path}")
    with open(segments_path, 'rb') as f:
        all_segments = pickle.load(f)
    
    print(f"Loading labels from: {labels_path}")
    with open(labels_path, 'rb') as f:
        all_labels = pickle.load(f)
    
    print(f"Loaded {len(all_segments)} songs")
    print(f"Loaded {len(all_labels)} label sets")
    
    return prepare_padded_training_data(all_segments, all_labels)

def prepare_padded_training_data(all_segments, all_labels, max_meters=201, max_frames=300, 
                                n_features=15, test_size=0.2, random_state=42):
    """
    Prepare data in the original CRNN format (padded songs instead of individual segments)
    
    This maintains the original song-level structure with padding
    """
    print("Preparing padded training data (song-level)...")
    
    X_songs = []
    y_songs = []
    
    for song_segments, song_labels in zip(all_segments, all_labels):
        # Pad song to max_meters
        padded_song = np.zeros((max_meters, max_frames, n_features))
        padded_labels = np.full((max_meters,), -1)  # Use -1 for padding
        
        # Fill in actual data
        n_segments = min(len(song_segments), max_meters)
        for i in range(n_segments):
            segment = song_segments[i]
            # Handle segment padding/truncation to max_frames
            if len(segment) <= max_frames:
                padded_song[i, :len(segment), :] = segment
            else:
                # Sample frames evenly if too long
                indices = np.linspace(0, len(segment)-1, max_frames, dtype=int)
                padded_song[i, :, :] = segment[indices, :]
            
            padded_labels[i] = song_labels[i]
        
        X_songs.append(padded_song)
        y_songs.append(padded_labels)
    
    X_songs = np.array(X_songs)
    y_songs = np.array(y_songs)
    
    print(f"Prepared {len(X_songs)} songs")
    print(f"Song shape: {X_songs.shape}")
    print(f"Labels shape: {y_songs.shape}")
    
    # Convert labels to categorical (handling padding)
    num_classes = len(np.unique(y_songs[y_songs >= 0]))  # Exclude padding (-1)
    y_categorical = np.full((*y_songs.shape, num_classes), 0.0)
    
    # Only convert non-padded labels
    mask = y_songs >= 0
    valid_labels = y_songs[mask]
    y_categorical[mask] = tf.keras.utils.to_categorical(valid_labels, num_classes=num_classes)
    # Set padded positions to -1 for masking
    y_categorical[~mask] = -1.0
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X_songs, y_categorical, test_size=test_size, random_state=random_state
    )
    
    print(f"Training songs: {len(X_train)}")
    print(f"Validation songs: {len(X_val)}")
    print(f"Classes: {num_classes}")
    
    return X_train, X_val, y_train, y_val, num_classes

# Architecture configurations for testing
ARCHITECTURE_CONFIGS = {
    'baseline': {
        'architecture_name': 'baseline',
        'num_classes': 4,
        'epochs': 10,
        'batch_size': 1,
        'learning_rate': 0.001,
        'early_stopping_patience': 5,
        'class_weights': None
    },
    
    'option1_dense_layers': {
        'architecture_name': 'option1_dense_layers',
        'num_classes': 4,
        'epochs': 10,
        'batch_size': 1,
        'learning_rate': 0.001,
        'early_stopping_patience': 5,
        'class_weights': None
    },
    
    'option2_lstm_512': {
        'architecture_name': 'option2_larger_lstm',
        'num_classes': 4,
        'lstm_units': 512,
        'epochs': 10,
        'batch_size': 1,
        'learning_rate': 0.001,
        'early_stopping_patience': 5,
        'class_weights': None
    },
    
    'option2_lstm_768': {
        'architecture_name': 'option2_larger_lstm',
        'num_classes': 4,
        'lstm_units': 768,
        'epochs': 10,
        'batch_size': 1,
        'learning_rate': 0.0005,  # Lower LR for larger model
        'early_stopping_patience': 5,
        'class_weights': None
    },
    
    'option3_stacked_lstm': {
        'architecture_name': 'option3_stacked_lstm',
        'num_classes': 4,
        'epochs': 12,
        'batch_size': 1,
        'learning_rate': 0.0005,
        'early_stopping_patience': 5,
        'class_weights': None
    },
    
    'option4_combination': {
        'architecture_name': 'option4_combination',
        'num_classes': 4,
        'epochs': 15,
        'batch_size': 1,
        'learning_rate': 0.0005,
        'early_stopping_patience': 7,
        'class_weights': None
    }
}

# Notebook-friendly functions with pickle support
def load_and_test_architecture(architecture_name, segments_path, labels_path, 
                             base_model_path, class_names=['O', 'A', 'B', 'C']):
    """
    Load pickle data and test architecture in one function - perfect for notebooks
    
    Args:
        architecture_name: Which architecture to test
        segments_path: Path to pickled segments file
        labels_path: Path to pickled labels file  
        base_model_path: Path to base CRNN model
        class_names: List of class names
    
    Usage:
        result = load_and_test_architecture('option4_combination', 
                                          'audio-segments-2024.pkl', 
                                          'all_labels-2024.pkl',
                                          'best_model_V3.h5')
    """
    print(f"🔧 Loading data and testing {architecture_name}...")
    
    # Load and prepare data
    X_train, X_val, y_train, y_val, num_classes = load_pickle_data(segments_path, labels_path)
    
    # Update class names if needed
    if num_classes != len(class_names):
        class_names = [f"Class_{i}" for i in range(num_classes)]
    
    # Now test the architecture
    return quick_test_architecture(architecture_name, X_train, y_train, X_val, y_val, 
                                 base_model_path, class_names, num_classes)

def quick_test_architecture(architecture_name, X_train, y_train, X_val, y_val, 
                          base_model_path, class_names=['O', 'A', 'B', 'C'], num_classes=4):
    """
    Quick test of a single architecture - perfect for notebook cells
    
    Usage in notebook:
    results = quick_test_architecture('option4_combination', X_train, y_train, X_val, y_val, 'best_model_V3.h5')
    """
    
    if architecture_name not in ARCHITECTURE_CONFIGS:
        print(f"Available architectures: {list(ARCHITECTURE_CONFIGS.keys())}")
        return None
    
    config = ARCHITECTURE_CONFIGS[architecture_name].copy()
    config['num_classes'] = num_classes  # Update for actual number of classes
    tester = CRNNArchitectureTester(base_model_path)
    
    # Setup experiment
    wandb_config = tester.setup_experiment(config)
    
    try:
        # Build model
        model = tester.modify_pretrained_model(wandb_config)
        if model is None:
            return None
            
        print(f"Model created successfully!")
        print(f"Total params: {model.count_params():,}")
        print(f"Trainable params: {sum([tf.keras.utils.count_params(w) for w in model.trainable_weights]):,}")
        
        # Train
        print(f"\\nStarting training for {architecture_name}...")
        history, model_save_path = tester.train_with_wandb(model, X_train, y_train, X_val, y_val, wandb_config)
        
        # Detailed evaluation with precision/recall
        print(f"\\n📊 Running detailed evaluation...")
        detailed_metrics = tester.evaluate_detailed_performance(model, X_val, y_val, class_names)
        
        # Get best metrics from training
        best_val_acc = max(history.history['val_masked_accuracy'])
        final_val_acc = history.history['val_masked_accuracy'][-1]
        
        results = {
            'architecture': architecture_name,
            'best_val_accuracy': best_val_acc,
            'final_val_accuracy': final_val_acc,
            'epochs_trained': len(history.history['loss']),
            'detailed_metrics': detailed_metrics,
            'model': model,
            'history': history,
            'wandb_run_id': wandb.run.id,
            'model_save_path': model_save_path
        }
        
        print(f"\\n✅ {architecture_name} completed!")
        print(f"Best val accuracy: {best_val_acc:.4f}")
        print(f"Final val accuracy: {final_val_acc:.4f}")
        print(f"Macro F1: {detailed_metrics.get('f1_macro', 0):.4f}")
        print(f"💾 Model saved: {model_save_path}")
        
        return results
        
    except Exception as e:
        print(f"❌ {architecture_name} failed: {e}")
        return None
    finally:
        wandb.finish()

def compare_architectures(architectures_to_test, X_train, y_train, X_val, y_val, base_model_path):
    """
    Compare multiple architectures
    
    Usage in notebook:
    results = compare_architectures(['baseline', 'option1_dense_layers', 'option4_combination'], 
                                  X_train, y_train, X_val, y_val, 'best_model_V3.h5')
    """
    results = {}
    
    for arch_name in architectures_to_test:
        print(f"\\n{'='*50}")
        print(f"Testing: {arch_name}")
        print(f"{'='*50}")
        
        result = quick_test_architecture(arch_name, X_train, y_train, X_val, y_val, base_model_path)
        if result:
            results[arch_name] = result
            
    # Summary
    print(f"\\n{'='*50}")
    print("RESULTS SUMMARY")
    print(f"{'='*50}")
    
    for name, result in results.items():
        print(f"{name:25s}: {result['best_val_accuracy']:.4f} (best) | {result['final_val_accuracy']:.4f} (final)")
    
    return results

# Example notebook usage
if __name__ == "__main__":
    print("Architecture tester ready!")
    print("\\nUsage in notebook:")
    print("1. Load your data: X_train, y_train, X_val, y_val")
    print("2. Test single architecture:")
    print("   result = quick_test_architecture('option4_combination', X_train, y_train, X_val, y_val, 'best_model_V3.h5')")
    print("3. Compare multiple:")
    print("   results = compare_architectures(['baseline', 'option1_dense_layers'], X_train, y_train, X_val, y_val, 'best_model_V3.h5')")
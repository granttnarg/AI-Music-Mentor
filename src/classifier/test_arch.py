import wandb
import tensorflow as tf
import numpy as np
from datetime import datetime
import os

class EnergyClassifierExperiment:
    def __init__(self, project_name="energy-classification"):
        """
        Initialize W&B experiment tracking
        """
        self.project_name = project_name

    def setup_experiment(self, experiment_config):
        """
        Initialize a new W&B run with configuration

        experiment_config should include:
        - architecture_name: string describing the model architecture
        - num_classes: 3 or 4
        - dataset_version: which pickle file you're using
        - class_weights: dict of class weights or None
        - other hyperparameters
        """

        # Initialize wandb run
        wandb.init(
            project=self.project_name,
            config=experiment_config,
            name=f"{experiment_config['architecture_name']}_{experiment_config['num_classes']}class",
            tags=[
                f"{experiment_config['num_classes']}-class",
                experiment_config['dataset_version'],
                experiment_config['architecture_name']
            ]
        )

        return wandb.config

    def build_model_from_config(self, config, base_model_path):
        """
        Build model based on W&B config
        """
        # Load base model
        base_model = tf.keras.models.load_model(base_model_path)

        # Get base features
        if 'feature_layer_name' in config:
            base_output = base_model.get_layer(config.feature_layer_name).output
        else:
            # Default: remove last few layers
            base_output = base_model.layers[-3].output

        # Build head based on architecture config
        x = base_output

        if config.architecture_name == 'simple':
            x = tf.keras.layers.Dense(config.num_classes, activation='softmax', name='predictions')(x)

        elif config.architecture_name == 'deep_head':
            for i, units in enumerate(config.head_layers):
                x = tf.keras.layers.Dense(units, activation='relu', name=f'dense_{i}')(x)
                if config.use_batch_norm:
                    x = tf.keras.layers.BatchNormalization(name=f'bn_{i}')(x)
                if config.dropout_rate > 0:
                    x = tf.keras.layers.Dropout(config.dropout_rate, name=f'dropout_{i}')(x)

            x = tf.keras.layers.Dense(config.num_classes, activation='softmax', name='predictions')(x)

        elif config.architecture_name == 'attention_head':
            # Add attention mechanism
            x = tf.keras.layers.Dense(config.attention_dim, activation='relu')(x)

            # Self-attention (simplified)
            attention_weights = tf.keras.layers.Dense(1, activation='softmax')(x)
            x = tf.keras.layers.Multiply()([x, attention_weights])

            x = tf.keras.layers.Dense(config.num_classes, activation='softmax', name='predictions')(x)

        elif config.architecture_name == 'residual_head':
            # Residual connections in head
            input_dim = x.shape[-1]

            # First residual block
            residual = x
            x = tf.keras.layers.Dense(config.hidden_dim, activation='relu')(x)
            x = tf.keras.layers.Dense(input_dim, activation='relu')(x)
            x = tf.keras.layers.Add()([x, residual])  # Residual connection

            # Final prediction
            x = tf.keras.layers.Dense(config.num_classes, activation='softmax', name='predictions')(x)

        # Create model
        model = tf.keras.Model(inputs=base_model.input, outputs=x, name=config.architecture_name)

        # Compile
        optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)

        model.compile(
            optimizer=optimizer,
            loss=config.loss_function,  # Your masked_categorical_crossentropy
            metrics=config.metrics  # Your masked_accuracy
        )

        # Log model architecture
        wandb.config.update({
            'total_params': model.count_params(),
            'trainable_params': sum([tf.keras.utils.count_params(w) for w in model.trainable_weights]),
            'model_layers': len(model.layers)
        })

        return model

    def train_with_wandb(self, model, X_train, y_train, X_val, y_val, config):
        """
        Train model with W&B logging
        """

        # Create W&B callback
        wandb_callback = wandb.keras.WandbCallback(
            monitor='val_masked_accuracy',
            mode='max',
            save_model=True,  # Save best model to W&B
            save_graph=True
        )

        # Other callbacks
        callbacks = [
            wandb_callback,
            tf.keras.callbacks.EarlyStopping(
                monitor='val_masked_accuracy',
                patience=config.early_stopping_patience,
                restore_best_weights=True,
                mode='max'
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6
            )
        ]

        # Train
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=config.epochs,
            batch_size=config.batch_size,
            class_weight=config.class_weights if hasattr(config, 'class_weights') else None,
            callbacks=callbacks,
            verbose=1
        )

        # Log final metrics
        final_metrics = {
            'final_train_acc': history.history['masked_accuracy'][-1],
            'final_val_acc': history.history['val_masked_accuracy'][-1],
            'final_train_loss': history.history['loss'][-1],
            'final_val_loss': history.history['val_loss'][-1],
            'best_val_acc': max(history.history['val_masked_accuracy']),
            'epochs_trained': len(history.history['loss'])
        }

        wandb.log(final_metrics)

        return history

    def evaluate_detailed_performance(self, model, X_test, y_test, class_names):
        """
        Detailed evaluation with per-class metrics logged to W&B
        """
        from sklearn.metrics import classification_report, confusion_matrix

        # Get predictions
        y_pred_proba = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=-1)
        y_true = np.argmax(y_test, axis=-1)

        # Flatten for sequence data
        y_pred_flat = y_pred.flatten()
        y_true_flat = y_true.flatten()

        # Remove masked values if using masking
        mask = y_true_flat >= 0  # Assuming -1 or similar for masked
        y_pred_flat = y_pred_flat[mask]
        y_true_flat = y_true_flat[mask]

        # Classification report
        report = classification_report(y_true_flat, y_pred_flat,
                                     target_names=class_names,
                                     output_dict=True)

        # Log per-class metrics
        class_metrics = {}
        for i, class_name in enumerate(class_names):
            if class_name in report:
                class_metrics.update({
                    f'{class_name}_precision': report[class_name]['precision'],
                    f'{class_name}_recall': report[class_name]['recall'],
                    f'{class_name}_f1': report[class_name]['f1-score'],
                    f'{class_name}_support': report[class_name]['support']
                })

        # Overall metrics
        class_metrics.update({
            'macro_avg_f1': report['macro avg']['f1-score'],
            'weighted_avg_f1': report['weighted avg']['f1-score'],
            'accuracy': report['accuracy']
        })

        wandb.log(class_metrics)

        # Confusion matrix
        cm = confusion_matrix(y_true_flat, y_pred_flat)
        wandb.log({
            "confusion_matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=y_true_flat,
                preds=y_pred_flat,
                class_names=class_names
            )
        })

        return class_metrics

# Configuration templates for different experiments
EXPERIMENT_CONFIGS = {
    '4class_simple': {
        'architecture_name': 'simple',
        'num_classes': 4,
        'dataset_version': '4class_augmented',
        'epochs': 15,
        'batch_size': 1,
        'learning_rate': 0.001,
        'class_weights': None,
        'loss_function': 'masked_categorical_crossentropy',
        'metrics': ['masked_accuracy'],
        'early_stopping_patience': 5
    },

    '4class_deep_head': {
        'architecture_name': 'deep_head',
        'num_classes': 4,
        'dataset_version': '4class_augmented',
        'head_layers': [128, 64],
        'use_batch_norm': True,
        'dropout_rate': 0.3,
        'epochs': 15,
        'batch_size': 1,
        'learning_rate': 0.001,
        'class_weights': {0: 1.5, 1: 2.0, 2: 0.7, 3: 1.0},
        'loss_function': 'masked_categorical_crossentropy',
        'metrics': ['masked_accuracy'],
        'early_stopping_patience': 5
    },

    # '3class_attention': {
    #     'architecture_name': 'attention_head',
    #     'num_classes': 3,
    #     'dataset_version': '3class_augmented',
    #     'attention_dim': 64,
    #     'epochs': 15,
    #     'batch_size': 1,
    #     'learning_rate': 0.001,
    #     'class_weights': {0: 0.8, 1: 4.0, 2: 1.5},
    #     'loss_function': 'masked_categorical_crossentropy',
    #     'metrics': ['masked_accuracy'],
    #     'early_stopping_patience': 5
    # }
}

# Usage example
def run_experiment_suite():
    """
    Run a suite of experiments with W&B tracking
    """
    experiment = EnergyClassifierExperiment("dj-energy-classification")

    # Load your pickled data
    X_train, X_val, y_train, y_val = load_and_split_data("energy_4class_features.pkl")

    results = {}

    for experiment_name, config in EXPERIMENT_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"Running experiment: {experiment_name}")
        print(f"{'='*60}")

        # Setup experiment
        wandb_config = experiment.setup_experiment(config)

        try:
            # Build model
            model = experiment.build_model_from_config(wandb_config, 'base_model.h5')

            # Train
            history = experiment.train_with_wandb(model, X_train, y_train, X_val, y_val, wandb_config)

            # Detailed evaluation
            class_names = ['O', 'A', 'B', 'C'] if config['num_classes'] == 4 else ['A', 'B', 'C']
            metrics = experiment.evaluate_detailed_performance(model, X_val, y_val, class_names)

            results[experiment_name] = {
                'config': config,
                'metrics': metrics,
                'wandb_run_id': wandb.run.id
            }

            print(f"Experiment {experiment_name} completed!")
            print(f"Best val accuracy: {metrics.get('accuracy', 'N/A'):.4f}")

        except Exception as e:
            print(f"Experiment {experiment_name} failed: {e}")

        finally:
            wandb.finish()

    return results

if __name__ == "__main__":
    # Run all experiments
    results = run_experiment_suite()

    # Print summary
    print("\n" + "="*60)
    print("EXPERIMENT SUITE RESULTS")
    print("="*60)

    for name, result in results.items():
        acc = result['metrics'].get('accuracy', 0)
        print(f"{name:20s}: {acc:.4f}")
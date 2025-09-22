# Energy Classification Model

## Model Details
- **Created**: 2025-09-20 12:57
- **Type**: 4-class energy level classifier
- **Classes**: ['O', 'A', 'B', 'C']
- **Segment Length**: 5 seconds
- **Training Data**: 25 songs + 4x tempo augmentation

## Performance (Latest Run)
- **Overall Validation Accuracy**: ~53.5%
- **O (Other)**: Precision=0.707, Recall=0.629, F1=0.665
- **A (High Energy)**: Precision=0.441, Recall=0.519, F1=0.477
- **B (Breakdown)**: Precision=0.455, Recall=0.383, F1=0.416
- **C (Low Energy)**: Precision=0.664, Recall=0.755, F1=0.707

## Key Features
- Catches ~52% of high energy sections (significant improvement from initial ~28%)
- 5-second temporal segments provide good balance of stability and precision
- Class weights used to improve A-class recall

## Usage
```python
from load_model import load_energy_classifier
model, config, class_weights = load_energy_classifier()
```

## Files
- `model.h5`: Trained Keras model
- `config.json`: Model configuration and metadata
- `class_weights.json`: Training class weights
- `load_model.py`: Simple loading script
- `README.md`: This file

# Energy Classification Model

## Model Details
- **Created**: 2025-09-21 09:30
- **Type**: 3-class energy level classifier
- **Classes**: ['A', 'B', 'C']
- **Segment Length**: 5 seconds
- **Training Data**: 25 songs + 4x tempo augmentation

## Performance (Latest Run)


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

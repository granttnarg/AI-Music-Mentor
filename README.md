# AI Music Mentor - Work in Progress

Upload your unfinished techno tracks and get AI-powered arrangement feedback to help you finish them.

## Setup

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

1. Clone the repository:

```bash
git clone https://github.com/granttnarg/AI-Music-Mentor.git
cd AI-Music-Mentor
```

2. Install dependencies:

```bash
uv sync
```

3. Run the Streamlit app:

```bash
uv run streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`

### Development

Format code:

```bash
uv run black .
```

Run tests:

```bash
uv run pytest
```

### Project Structure

```
├── app.py              # Streamlit dashboard
├── src/
│   ├── audio_features.py   # Audio feature extraction
│   ├── rag.py             # RAG system for feedback
│   └── preprocessing.py   # Data preprocessing
├── data/
│   ├── raw/              # Training audio files
│   ├── processed/        # Extracted features
│   └── feedback/         # Manual feedback data
├── models/               # Trained models
└── tests/               # Unit tests
```

## How it works

1. **Upload** your unfinished MP3 track
2. **Select** feedback style from dropdown
3. **Enter** specific questions or areas of focus
4. **Get** personalized arrangement advice based on similar finished tracks

Built with RAG (Retrieval-Augmented Generation) to provide contextual feedback from experienced producers.

# AI Music Mentor - 🚧 Work in Progress - 3 Week MVP for Data Science Retreat

An AI-powered music production feedback tool that analyzes your unfinished techno tracks and provides personalized arrangement advice using RAG (Retrieval-Augmented Generation) technology and deep learning arrangement classification.

## Setup

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager
- PostgreSQL database
- [Ollama](https://ollama.ai/) with llama3.2 model

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

3. Set up environment variables:

```bash
cp .env.example .env
# Edit .env with your secure database credentials and connection details
```

4. Start Docker services (PostgreSQL + PgAdmin):

```bash
# Start database and admin interface
docker-compose up -d

# Check containers are running
docker-compose ps

# View logs if needed
docker-compose logs
```

5. Set up database:

```bash
uv run python -m admin  # Use admin interface to add DB examples for the RAG system to compare against.
```

6. Start Ollama and pull the model:

```bash
ollama serve
ollama pull llama3.2:latest
```

7. Run the Streamlit app:

```bash
uv run streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`

**Note**: Docker services must be running before starting the Streamlit app as it requires the PostgreSQL database connection.

### Running Modules

When running specific modules, use the -m flag so uv can resolve imports from the root directory:

```bash
uv run python -m services.audio_rag
```

### Development

Format code:

```bash
uv run black .
```

Run tests:

```bash
uv run pytest
```

### Docker Management

Stop services:

```bash
docker-compose down
```

Rebuild after changes:

```bash
docker-compose down -v  # Remove volumes to wipe data
docker-compose up -d    # Rebuild with fresh data
```

Access database directly:

- **PostgreSQL**: `localhost:5434`
- **PgAdmin**: `http://localhost:8080` (use credentials from .env)

### Project Structure

```
├── app.py                  # Main Streamlit User dashboard
├── admin.py                # Streamlit Admin dasboard to popular data via UI
├── main.py                 # Main App file
├── pyproject.toml          # Project dependencies and configuration
├── services/
│   └── audio_rag.py        # RAG system with LLM integration
├── src/
│   ├── audio_features.py   # Audio feature extraction using Librosa
│   └── classifier/         # Arrangement classification system
│       ├── arrangement_classifier.py      # CRNN model for arrangement analysis
│       ├── arrangement_postprocessing.py  # Pattern smoothing and analysis
│       └── feature_extraction.py         # Audio preprocessing for classification
├── db/
│   ├── db.py              # Database connection and setup
│   ├── models.py          # SQLAlchemy data models
│   └── operations.py      # Database operations and queries
├── data/
│   ├── raw/               # Raw training audio files
│   ├── processed/         # Processed feature data
│   ├── test/              # Test audio file
│   ├── uploads/           # User uploaded files and session data
│   └── batch_import/      # Training examples for batch processing
├── notebooks/             # Development and analysis notebooks
├── scripts/
│   └── batch_import.py    # Batch processing for training examples
├── models/
│   └── arrangement_classifier/  # Pre-trained arrangement classification models
├── visualizations/         # Example visualization outputs
└── tests/                # Unit tests
```

## How it works

1. **Upload** your unfinished MP3 track and a reference through the Streamlit interface
2. **Select** your music genre (deep techno, hard techno, house, etc.)
3. **Enter** specific questions or areas you want feedback on
4. **Get** AI-powered arrangement advice using:
   - Audio feature extraction with Librosa
   - Deep learning arrangement classification (CRNN model)
   - RAG system that retrieves similar tracks from database
   - LLM-generated personalized feedback using Ollama/LangChain

![Example Arrangement Analysis](visualizatios/example-viz-4classes.png)
_Example: Arrangement classification showing different sections (O=Other, A=Medium Energy, B=High Energy, C=Breakdown)_

## Features

- **Audio Analysis**: Extracts musical features using Librosa (tempo, key, spectral features, etc.)
- **Arrangement Classification**: Deep learning model that identifies track sections (intro/outro, medium energy, high energy, breakdown)
- **Pattern Recognition**: Compressed arrangement patterns (e.g., "1A-4O-8C-3B") for quick structure analysis
- **RAG System**: Retrieval-augmented generation for contextual feedback
- **Database Storage**: PostgreSQL/pgvector backend for tracks, features, and user feedback
- **Batch Import**: Process multiple training examples with automatic arrangement analysis
- **Admin Interface**: Tools for managing training data and user feedback
- **LangChain/Smith Integration**: Observability and tracing for LLM interactions
- **Genre Support**: Specialized for electronic music genres (techno, house, electro, etc.)

## License

Copyright (c) 2025 Grant Thomas
All rights reserved.

This code is provided for reference and learning purposes only.
It may not be copied, modified, or used in any project,
commercial or otherwise, without explicit permission.

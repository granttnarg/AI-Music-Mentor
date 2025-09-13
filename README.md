# AI Music Mentor - 🚧 Work in Progress - 3 Week MVP for Data Science Retreat

An AI-powered music production feedback tool that analyzes your unfinished techno tracks and provides personalized arrangement advice using RAG (Retrieval-Augmented Generation) technology.

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
# Edit .env to add your database connection URL
```

4. Set up database:

```bash
uv run python -m admin  # Use admin interface to manage database
```

5. Start Ollama and pull the model:

```bash
ollama serve
ollama pull llama3.2:latest
```

6. Run the Streamlit app:

```bash
uv run streamlit run app.py
```

### Running Modules

When running specific modules, use the -m flag so uv can resolve imports from the root directory:

```bash
uv run python -m services.audio_rag
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
├── app.py                  # Main Streamlit dashboard
├── admin.py                # Database administration interface
├── main.py                 # Audio feature extraction testing
├── pyproject.toml          # Project dependencies and configuration
├── services/
│   └── audio_rag.py        # RAG system with LLM integration
├── src/
│   └── audio_features.py   # Audio feature extraction using Librosa
├── db/
│   ├── db.py              # Database connection and setup
│   ├── models.py          # SQLAlchemy data models
│   └── operations.py      # Database operations and queries
├── data/
│   ├── raw/               # Raw training audio files
│   ├── processed/         # Processed feature data
│   └── test/             # Test audio files
├── uploads/               # User uploaded files and session data
├── notebooks/             # Development and analysis notebooks
└── tests/                # Unit tests
```

## How it works

1. **Upload** your unfinished MP3 track through the Streamlit interface
2. **Select** your music genre (deep techno, hard techno, house, etc.)
3. **Enter** specific questions or areas you want feedback on
4. **Get** AI-powered arrangement advice using:
   - Audio feature extraction with Librosa
   - RAG system that retrieves similar tracks from database
   - LLM-generated personalized feedback using Ollama/LangChain

## Features

- **Audio Analysis**: Extracts musical features using Librosa (tempo, key, spectral features, etc.)
- **RAG System**: Retrieval-augmented generation for contextual feedback
- **Database Storage**: PostgreSQL backend for tracks, features, and user feedback
- **Admin Interface**: Tools for managing training data and user feedback
- **LangSmith Integration**: Observability and tracing for LLM interactions
- **Genre Support**: Specialized for electronic music genres (techno, house, electro, etc.)

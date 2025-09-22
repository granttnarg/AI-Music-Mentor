import logging
import os
from dotenv import load_dotenv


def setup_logging(level=logging.INFO):
    """Configure logging for the application"""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,  # Override any existing config
    )

    # Setup predictions logger for debugging
    predictions_logger = logging.getLogger('predictions')
    predictions_logger.setLevel(logging.INFO)

    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)

    # File handler for predictions
    predictions_handler = logging.FileHandler('logs/predictions.log')
    predictions_handler.setLevel(logging.INFO)
    predictions_formatter = logging.Formatter('%(asctime)s - %(message)s')
    predictions_handler.setFormatter(predictions_formatter)

    # Avoid duplicate handlers
    if not predictions_logger.handlers:
        predictions_logger.addHandler(predictions_handler)

    predictions_logger.propagate = False  # Don't propagate to root logger


def setup_database():
    """Initialize database schema if needed"""
    from db.db import AudioRAGDatabase

    connection_url = os.getenv("DB_CONNECTION_URL")
    if not connection_url or "your_password" in connection_url:
        logging.warning("DB_CONNECTION_URL not configured, skipping database setup")
        return None

    try:
        db = AudioRAGDatabase(connection_url)
        db.setup_database()
        logging.info("Database schema verified/created")
        return db
    except Exception as e:
        logging.error(f"Database setup failed: {e}")
        return None


def setup_environment():
    """Load environment variables"""
    load_dotenv()


def init_app():
    """Initialize common app setup - call from entry points"""
    setup_environment()
    setup_logging()
    setup_database()

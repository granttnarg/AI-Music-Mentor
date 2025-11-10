import os
from db.db import AudioRAGDatabase
from db.operations import AudioRAGOperations


def get_database() -> AudioRAGOperations:
    """
    Initialize and return database operations instance.

    Returns:
        AudioRAGOperations: Database operations instance with connection pool
    """
    connection_url = os.getenv(
        "DB_CONNECTION_URL", "postgresql://postgres:<ADD_TOENV_FILE>"
    )
    db = AudioRAGDatabase(connection_url)
    return AudioRAGOperations(db)

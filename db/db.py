from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from typing import List
from .models import Track, Base

# from .db_models import TrainingExample, Feedback, UserUpload


class AudioRAGDatabase:
    def __init__(self, connection_string: str):
        self.engine = create_engine(connection_string)
        self.SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=self.engine
        )

    def setup_database(self):
        """Create all tables"""
        # Enable pgvector extension (raw SQL needed for this)
        with self.engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            conn.commit()

        # Create all tables
        Base.metadata.create_all(bind=self.engine)
        print("✓ Database schema created!")

    def reset_database(self):
        """Drop all tables"""
        Base.metadata.drop_all(bind=self.engine)
        print("✓ All tables dropped!")

    def get_session(self):
        return self.SessionLocal()

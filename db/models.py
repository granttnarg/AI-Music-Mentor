from sqlalchemy import Column, Integer, String, Float, Text, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector
from datetime import datetime

Base = declarative_base()


class Track(Base):
    __tablename__ = "tracks"

    id = Column(Integer, primary_key=True)
    file_path = Column(String, unique=True, nullable=False)
    duration = Column(Float)
    sample_rate = Column(Integer)
    global_embedding = Column(Vector(19))
    processed_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.now)

    # Relationships
    example_tracks = relationship(
        "TrainingExample", foreign_keys="TrainingExample.example_track_id"
    )
    reference_tracks = relationship(
        "TrainingExample", foreign_keys="TrainingExample.reference_track_id"
    )
    input_uploads = relationship("UserUpload", foreign_keys="UserUpload.input_track_id")
    reference_uploads = relationship("UserUpload", foreign_keys="UserUpload.reference_track_id")


class TrainingExample(Base):
    __tablename__ = "training_examples"

    id = Column(Integer, primary_key=True)
    example_track_id = Column(Integer, ForeignKey("tracks.id"), nullable=False)
    reference_track_id = Column(Integer, ForeignKey("tracks.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.now)

    # Relationships
    example_track = relationship("Track", foreign_keys=[example_track_id])
    reference_track = relationship("Track", foreign_keys=[reference_track_id])
    feedback_items = relationship("Feedback", back_populates="training_example")


class Feedback(Base):
    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True)
    training_example_id = Column(
        Integer, ForeignKey("training_examples.id"), nullable=False
    )
    feedback_type = Column(
        String, nullable=False
    )  # 'rhythm', 'eq', 'global', 'arrangement', 'energy'
    feedback_text = Column(Text)
    created_at = Column(DateTime, default=datetime.now)

    # Relationships
    training_example = relationship("TrainingExample", back_populates="feedback_items")


class UserUpload(Base):
    __tablename__ = "user_uploads"

    id = Column(Integer, primary_key=True)
    input_track_id = Column(Integer, ForeignKey("tracks.id"), nullable=False)
    reference_track_id = Column(Integer, ForeignKey('tracks.id'), nullable=False) # we dont need this yet.

    user_prompt = Column(Text)
    stage = Column(String)
    genre = Column(String)
    created_at = Column(DateTime, default=datetime.now)

    # Relationships
    input_track = relationship("Track", foreign_keys=[input_track_id])
    reference_track = relationship("Track", foreign_keys=[reference_track_id])

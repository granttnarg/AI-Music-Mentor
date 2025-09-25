#!/usr/bin/env python3
"""
Generate visualizations for all tracks in the database.

This script loops through all tracks and generates cached waveform and timeline visualizations
using the SongVisualizerService.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import init_app
from db.db import AudioRAGDatabase
from db.models import Track
from services.song_visualizer_service import SongVisualizerService
from sqlalchemy import text

# Initialize logging and environment
init_app()

import logging

logger = logging.getLogger(__name__)


def generate_all_visualizations(db_connection_url: str):
    """Generate visualizations for all tracks in the database."""
    
    # Initialize services
    db = AudioRAGDatabase(db_connection_url)
    visualizer = SongVisualizerService()
    
    session = db.get_session()
    try:
        # Get all tracks from database
        tracks = session.query(Track).all()
        
        if not tracks:
            print("No tracks found in database")
            return
            
        print(f"Found {len(tracks)} tracks to process")
        
        successful = 0
        failed = 0
        
        for track in tracks:
            try:
                print(f"\n🎨 Processing track {track.id}: {Path(track.file_path).name}")
                
                # Check if file exists
                if not Path(track.file_path).exists():
                    print(f"❌ File not found: {track.file_path}")
                    failed += 1
                    continue
                
                # Get actual file duration (not truncated duration from database)
                print(f"   Loading audio to get actual duration...")
                import librosa
                y, sr = librosa.load(track.file_path, sr=None)
                actual_duration = len(y) / sr
                print(f"   Actual duration: {actual_duration:.1f}s (DB had: {track.duration:.1f}s)")
                
                # Create arrangement blocks using the proper helper function
                arrangement_blocks = None
                if track.raw_predictions and track.raw_confidence_scores:
                    try:
                        print(f"   Creating arrangement blocks from raw predictions...")
                        from src.classifier.arrangement_postprocessing import create_arrangement_blocks
                        import json
                        import numpy as np
                        
                        raw_predictions = json.loads(track.raw_predictions)
                        confidence_scores = json.loads(track.raw_confidence_scores)
                        
                        # Calculate segment duration based on actual track length
                        segment_duration = actual_duration / len(raw_predictions)
                        
                        arrangement_blocks = create_arrangement_blocks(
                            np.array(raw_predictions), 
                            ['O', 'A', 'B', 'C'],
                            segment_duration=segment_duration
                        )
                        
                        print(f"   Created {len(arrangement_blocks)} arrangement blocks with proper timing")
                        
                    except Exception as e:
                        print(f"   ⚠️  Failed to create arrangement blocks: {e}")
                        arrangement_blocks = None
                
                # Generate waveform visualization
                waveform_path = visualizer.plot_arrangement_waveform_cached(
                    track_id=track.id,
                    audio_path=track.file_path,
                    arrangement_blocks=arrangement_blocks,
                    title=f"Track {track.id}: {Path(track.file_path).name}"
                )
                
                # Update track with visualization paths  
                print(f"   Updating database for track {track.id}...")
                track.waveform_viz_path = waveform_path
                track.viz_generated_at = datetime.now()
                
                # Explicitly mark the track as modified and add to session
                session.add(track)
                session.flush()  # Ensure the update is pending
                print(f"   Database update flushed successfully")
                
                print(f"✅ Generated waveform: {waveform_path}")
                print(f"   Database will be updated with: {waveform_path}")
                successful += 1
                    
            except Exception as e:
                print(f"❌ Error processing track {track.id}: {e}")
                failed += 1
        
        # Commit all updates
        print(f"\n💾 Committing {successful} database updates...")
        session.commit()
        print(f"✅ Database commit successful!")
        
        print(f"\n🎉 Visualization generation complete!")
        print(f"   Successful: {successful}")
        print(f"   Failed: {failed}")
        print(f"   Total: {len(tracks)}")
        
    except Exception as e:
        print(f"❌ Database error: {e}")
        session.rollback()
    finally:
        session.close()


def main():
    """Main entry point."""
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    db_url = os.getenv("DB_CONNECTION_URL")
    if not db_url:
        print("❌ DB_CONNECTION_URL not found in environment")
        return 1
    
    print("🚀 Starting visualization generation for all tracks...")
    
    try:
        generate_all_visualizations(db_url)
        print("✅ All done!")
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
from db.operations import AudioRAGOperations, AudioRAGDatabase
from db.models import Track
import random

def create_fake_training_example(ops):
    """Create a fake training example using existing tracks"""

    # Get all tracks from the database
    session = ops.db.get_session()
    tracks = session.query(Track).all()
    session.close()

    if len(tracks) < 2:
        print("Error: Add more Tracks to DB to run this method - >= 2 entries")
        return None

    # Pick two random tracks
    example_track = random.choice(tracks)
    reference_track = random.choice([t for t in tracks if t.id != example_track.id])

    # Create some realistic feedback
    feedback_types = ['rhythm', 'eq', 'global', 'arrangement', 'energy']

    feedback_examples = {
        'rhythm': [
            'The kick pattern needs more variation and groove',
            'Hi-hats are too repetitive, add some swing',
            'The snare feels weak, needs more punch',
            'Percussion elements are cluttering the mix'
        ],
        'eq': [
            'Too much low-end muddying the mix',
            'High frequencies are harsh and fatiguing',
            'Needs more presence in the mid-range',
            'Bass and kick are fighting in the low-end'
        ],
        'global': [
            'Overall mix feels unbalanced',
            'Track lacks cohesion between sections',
            'Dynamics are too compressed',
            'Stereo image could be wider'
        ],
        'arrangement': [
            'Intro is too long and repetitive',
            'Breakdown section needs more tension',
            'Track needs a proper climax moment',
            'Transitions between sections are abrupt'
        ],
        'energy': [
            'Energy level stays flat throughout',
            'Needs more build-up to the drop',
            'Second half loses momentum',
            'Opening lacks impact and excitement'
        ]
    }

    # Generate 2-4 random feedback items
    num_feedback = random.randint(2, 4)
    selected_types = random.sample(feedback_types, num_feedback)

    feedback_items = []
    for feedback_type in selected_types:
        feedback_text = random.choice(feedback_examples[feedback_type])
        feedback_items.append({
            'feedback_type': feedback_type,
            'feedback_text': feedback_text
        })

    try:
        training_id = ops.add_training_example_for_mockdata(
            example_track_path=example_track.file_path,
            reference_track_path=reference_track.file_path,
            feedback_items=feedback_items
        )

        print(f"  Created training example {training_id}")
        print(f"   Example track: {example_track.file_path}")
        print(f"   Reference track: {reference_track.file_path}")
        print(f"   Feedback items:")
        for item in feedback_items:
            print(f"     - {item['feedback_type']}: {item['feedback_text']}")

        return training_id

    except Exception as e:
        print(f" ERROR: Failed to create training example: {e}")
        return None

if __name__ == "__main__":
    from dotenv import load_dotenv

    # Load environment variables from .env file
    load_dotenv()
    # Connect to database
    db = AudioRAGDatabase("postgresql://postgres:password@127.0.0.1:5434/audio_rag")
    ops = AudioRAGOperations(db)

    print("🎵 Creating fake training examples...")

    # Create 3 fake training examples
    for i in range(3):
        print(f"\n--- Training Example {i+1} ---")
        create_fake_training_example(ops)

    # Show summary
    session = ops.db.get_session()
    from db.models import TrainingExample, Feedback

    training_count = session.query(TrainingExample).count()
    feedback_count = session.query(Feedback).count()

    print(f"\n Database Summary:")
    print(f"   Training Examples: {training_count}")
    print(f"   Feedback Items: {feedback_count}")

    session.close()
    print("\n Done!")
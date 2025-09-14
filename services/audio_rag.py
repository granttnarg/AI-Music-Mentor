from typing import List, Dict, Any
from db.operations import AudioRAGOperations
from db.db import AudioRAGDatabase
from db.models import TrainingExample, Track, UserUpload, Feedback
import os

# LangChain imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

# LangSmith imports
from langsmith import traceable


class AudioRAG:
    def __init__(self, db: AudioRAGDatabase, llm_model: str = "llama3.2:latest"):
        self.db = db
        self.operations = AudioRAGOperations(db)
        self.llm_model = llm_model

        os.environ["LANGCHAIN_TRACING_V2"] = "true"

        # Initialize RAG components
        self.prompt = self.create_prompt_template()
        self.output_parser = StrOutputParser()
        self.llm = ChatOllama(
            model=llm_model, temperature=0.95, base_url="http://localhost:11434"
        )
        self.chain = self.prompt | self.llm | self.output_parser

    @traceable(name="retrieve_similar_examples")
    def retrieve_similar_examples(
        self, user_upload_id: int, k: int = 5, metric: str = "cosine"
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-k most similar training examples for a given user upload

        Args:
            user_upload_id: ID of the user upload
            k: Number of similar examples to return
            metric: Distance metric ("cosine" or "euclidean")

        Returns:
            List of dictionaries containing training example data and similarity info
        """
        session = self.db.get_session()

        try:
            # Get the user upload and its input track
            user_upload = (
                session.query(UserUpload)
                .filter(UserUpload.id == user_upload_id)
                .first()
            )
            if not user_upload:
                raise ValueError(f"User upload {user_upload_id} not found")

            input_track = (
                session.query(Track)
                .filter(Track.id == user_upload.input_track_id)
                .first()
            )
            if not input_track or input_track.global_embedding is None:
                raise ValueError(
                    f"Input track embedding not found for user upload {user_upload_id}"
                )

            # Use existing find_similar_tracks method to get similar tracks
            similar_tracks = self.operations.find_similar_tracks(
                embedding=list(input_track.global_embedding),
                metric=metric,
                limit=k * 3,  # Get more tracks since we'll filter for training examples
            )

            # Filter tracks that are part of training examples and get the training data
            results = []
            for track in similar_tracks:
                if len(results) >= k:
                    break

                # Find training examples where this track is the example track
                training_examples = (
                    session.query(TrainingExample)
                    .filter(TrainingExample.example_track_id == track.id)
                    .all()
                )

                for training_example in training_examples:
                    if len(results) >= k:
                        break

                    # Get reference track
                    reference_track = (
                        session.query(Track)
                        .filter(Track.id == training_example.reference_track_id)
                        .first()
                    )

                    # Get feedback for this training example
                    feedback_items = (
                        session.query(Feedback)
                        .filter(Feedback.training_example_id == training_example.id)
                        .all()
                    )

                    result = {
                        "training_example_id": training_example.id,
                        "similarity_rank": len(results) + 1,
                        "example_track": {
                            "id": track.id,
                            "file_path": track.file_path,
                            "embedding": list(track.global_embedding),
                            "duration": track.duration,
                            "sample_rate": track.sample_rate,
                        },
                        "reference_track": (
                            {
                                "id": reference_track.id,
                                "file_path": reference_track.file_path,
                                "embedding": (
                                    list(reference_track.global_embedding)
                                    if reference_track.global_embedding is not None
                                    else None
                                ),
                                "duration": reference_track.duration,
                                "sample_rate": reference_track.sample_rate,
                            }
                            if reference_track
                            else None
                        ),
                        "feedback": [
                            {
                                "type": fb.feedback_type,
                                "text": fb.feedback_text,
                                "created_at": str(fb.created_at),
                            }
                            for fb in feedback_items
                        ],
                        "created_at": str(training_example.created_at),
                    }
                    results.append(result)

            # Create summary for LangSmith output tracking
            retrieval_summary = {
                "user_upload_id": user_upload_id,
                "k_requested": k,
                "k_found": len(results),
                "metric": metric,
                "user_genre": user_upload.genre if user_upload else None,
                "retrieved_tracks": [
                    {
                        "training_id": r["training_example_id"],
                        "track_name": r["example_track"]["file_path"].split("/")[-1],
                        "feedback_types": [fb["type"] for fb in r["feedback"]],
                    }
                    for r in results
                ],
            }

            # This will be captured in the trace output
            return results, user_upload, retrieval_summary

        except Exception as e:
            print(f"Error retrieving similar examples: {e}")
            raise
        finally:
            session.close()

    @traceable
    def format_examples_for_prompt(
        self, similar_examples: List[Dict[str, Any]], user_upload: UserUpload
    ) -> str:
        """
        Format retrieved similar examples into a structured string for the prompt
        Include user upload context (prompt stage, genre) and feedback examples
        """
        if not similar_examples:
            return "No similar examples found."

        # Start with user context
        context = f"User Upload Context:\n"
        context += f"  User Prompt Notes: {user_upload.user_prompt}\n"
        context += f"  Stage: {user_upload.stage}\n"
        context += f"  Genre: {user_upload.genre}\n\n"
        context += "Similar Examples:\n\n"

        formatted_examples = []

        for i, example in enumerate(similar_examples, 1):
            example_text = f"Example {i}:"

            # Add basic example track info
            example_track = example.get("example_track", {})
            example_text += f"\n  Track: {os.path.basename(example_track.get('file_path', 'Unknown'))}"
            example_text += f"\n  Duration: {example_track.get('duration', 'Unknown')}s"

            # Add feedback - this is the main learning content
            feedback_items = example.get("feedback", [])
            if feedback_items:
                example_text += "\n  Feedback:"
                for feedback in feedback_items:
                    example_text += f"\n    - {feedback.get('type', 'General')}: {feedback.get('text', 'No text')}"
            else:
                example_text += "\n  Feedback: No feedback available"

            formatted_examples.append(example_text)

        context += "\n\n".join(formatted_examples)

        # Add summary of example quality
        total_feedback_items = sum(
            len(ex.get("feedback", [])) for ex in similar_examples
        )
        context += f"\n\n[Retrieved {len(similar_examples)} examples with {total_feedback_items} total feedback items]"

        return context

    def create_prompt_template(self) -> ChatPromptTemplate:
        """
        Create a LangChain prompt template for music feedback generation
        """
        template = """You are an AI music mentor providing feedback on audio tracks.

        You have access to training examples showing how other tracks were improved. Each example contains:
        - An INPUT TRACK (unfinished, like the user's track)
        - A REFERENCE TRACK (finished version)
        - FEEDBACK explaining how to get from input to reference in the tone of one specific producer.

        Here are the relevant training examples:

        {examples}

        CRITICAL INSTRUCTIONS:
        - The user has uploaded a HALF-FINISHED track similar to the INPUT tracks in the examples above
        - You do NOT know what specific elements are already in the user's track
        - Use the feedback patterns from the examples to suggest what COULD BE ADDED or IMPROVED
        - Frame suggestions as "you could try..." or "consider adding..." rather than "your track has..." or "change your..."
        - If examples don't cover an aspect, say "Based on similar tracks at this stage, I'd need to hear more to provide specific feedback on [aspect]"

        User's Question: {question}
        User's Track Context: Half-finished track needing arrangement help

        Provide feedback in these categories:

        **RHYTHM:**
        Based on patterns from similar tracks, suggest rhythmic elements that could be developed or added...

        **ENERGY:**
        Based on patterns from similar tracks, suggest energy-building techniques that could be applied...

        Remember: You're suggesting potential improvements based on what worked for similar tracks, not describing what's already there."""

        return ChatPromptTemplate.from_template(template)

    def check_ollama_connection(self) -> bool:
        """
        Check if Ollama server is running and accessible
        """
        try:
            # Simple test to see if we can reach the LLM
            test_response = self.llm.invoke("Hello")
            return True
        except Exception as e:
            print(f"Ollama connection failed: {e}")
            print("Make sure Ollama is running with: ollama serve")
            print(f"And that the model '{self.llm_model}' is available")
            return False

    @traceable
    def generate_feedback(
        self, user_upload_id: int, question: str = "", k: int = 5
    ) -> str:
        """
        Complete RAG pipeline: retrieve, format, prompt, and generate feedback
        """
        # Retrieve similar examples
        similar_examples, user_upload, retrieval_info = self.retrieve_similar_examples(
            user_upload_id, k=k
        )

        # Format examples for prompt
        formatted_examples = self.format_examples_for_prompt(
            similar_examples, user_upload
        )

        # Prepare input for the chain
        chain_input = {
            "examples": formatted_examples,
            "question": (
                question
                if question
                else f"Please provide feedback on my {user_upload.genre} track."
            ),
        }

        # Generate feedback using the pre-initialized RAG chain
        try:
            feedback = self.chain.invoke(chain_input)
            return feedback
        except Exception as e:
            print(f"Error generating feedback with LLM: {e}")
            # Fallback to returning formatted prompt if LLM fails
            return self.prompt.format(**chain_input)


# Example usage
if __name__ == "__main__":
    from dotenv import load_dotenv
    from db.db import AudioRAGDatabase

    # Load environment variables from .env file
    load_dotenv()

    # Initialize database and RAG
    connection_url = os.getenv(
        "DB_CONNECTION_URL", "postgresql://postgres:<ADD_TOENV_FILE>"
    )
    db = AudioRAGDatabase(connection_url)
    rag = AudioRAG(db)

    # Test the complete RAG pipeline with user upload ID 1
    try:
        # Test retrieval and formatting
        similar_examples, user_upload = rag.retrieve_similar_examples(
            user_upload_id=1, k=3
        )
        formatted_examples = rag.format_examples_for_prompt(
            similar_examples, user_upload
        )
        print("=== Formatted Examples ===")
        print(formatted_examples)
        print("\n" + "=" * 50 + "\n")

        # Test complete feedback generation
        feedback = rag.generate_feedback(user_upload_id=1, k=3)
        print("=== Generated Feedback ===")
        print(feedback)

    except Exception as e:
        print(f"Error: {e}")

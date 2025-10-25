from typing import List, Dict, Any, Tuple, cast
from db.operations import AudioRAGOperations
from db.db import AudioRAGDatabase
from db.models import TrainingExample, Track, UserUpload, Feedback
from services.feature_comparison_service import FeatureComparisonService
from services.prompt_loader import PromptLoader
from services.rag_text_formatter import RagTextFormatter
import os

# LangChain imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

# LangSmith imports
from langsmith import traceable


class AudioRAG:
    def __init__(self, db: AudioRAGDatabase, llm_model: str = "qwen3:8b"):
        self.db = db
        self.operations = AudioRAGOperations(db)
        self.llm_model = llm_model

        os.environ["LANGCHAIN_TRACING_V2"] = "true"

        # Load prompts from YAML
        self.prompts = PromptLoader._load_prompts()

        # Initialize RAG components
        self.prompt = self.create_prompt_template()
        self.output_parser = StrOutputParser()

        base_url = os.environ["DEVELOPMENT_BASE_URL"]

        self.llm = ChatOllama(model=llm_model, temperature=0.4, base_url=base_url)
        self.chain = self.prompt | self.llm | self.output_parser

    @traceable(name="retrieve_similar_examples")
    def retrieve_similar_examples(
        self, user_upload_id: int, k: int = 5, metric: str = "cosine"
    ) -> Tuple[List[Dict[str, Any]], Any, Dict[str, Any]]:
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

            # Use the new optimized method that only searches tracks with training examples
            similar_tracks = self.operations.find_similar_tracks_with_training_examples(
                embedding=input_track.global_embedding.tolist(),
                metric=metric,
                limit=k,  # No need to multiply since we're pre-filtering
            )

            print(f"🔍 Found {len(similar_tracks)} tracks with training examples")

            if not similar_tracks:
                return (
                    [],
                    user_upload,
                    {
                        "user_upload_id": user_upload_id,
                        "k_requested": k,
                        "k_found": 0,
                        "metric": metric,
                        "user_genre": user_upload.genre if user_upload else None,
                        "retrieved_tracks": [],
                    },
                )

            # Build results from similar tracks (which already have training examples)
            results = self._build_training_example_results(similar_tracks, k, session)

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

            # Add similarity scores for debugging
            for i, result in enumerate(results):
                result["similarity_rank"] = i + 1

            # This will be captured in the trace output
            return results, user_upload, retrieval_summary

        except Exception as e:
            print(f"Error retrieving similar examples: {e}")
            raise
        finally:
            session.close()

    @traceable
    def format_examples_for_prompt(
        self,
        similar_examples: List[Dict[str, Any]],
        user_upload: UserUpload,
        question: str = "",
    ) -> str:
        """
        Format retrieved similar examples into a structured string for the prompt
        Include user upload context (prompt stage, genre) and feedback examples
        """
        if not similar_examples:
            return "No similar examples found."

        # Build user context section
        context = self._build_user_context(user_upload)

        # Rank and select most relevant feedback
        user_question = (
            question
            if question.strip()
            else (
                getattr(user_upload, "user_prompt_notes", "general feedback")
                or "general feedback"
            )
        )
        user_genre = getattr(user_upload, "genre", "electronic")
        ranked_feedback = self._rank_and_select_feedback(
            similar_examples, user_question, user_genre
        )

        # Format the final examples
        context += (
            f"Most Relevant Feedback (from {len(similar_examples)} similar tracks):\n\n"
        )

        formatted_examples = []
        for i, feedback in enumerate(ranked_feedback, 1):
            source_example = feedback.get("source_example", {})
            example_track = source_example.get("example_track", {})

            feedback_text = f"Relevant Example {i}:"
            feedback_text += f"\n  Source Track: {os.path.basename(example_track.get('file_path', 'Unknown'))}"
            feedback_text += (
                f"\n  Structure: {example_track.get('arrangement_pattern', 'Unknown')}"
            )
            feedback_text += f"\n  Feedback Type: {feedback.get('type', 'General')}"
            feedback_text += f"\n  Advice: {feedback.get('text', 'No text')}"

            formatted_examples.append(feedback_text)

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
        template = self.prompts.get("feedback_generation", {}).get("template", "")
        if not template:
            print(
                "Warning: Could not load feedback_generation template from YAML, using fallback"
            )
            template = "You are an AI music mentor. Provide feedback on: {question}"
        return ChatPromptTemplate.from_template(template)

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

        # Check retrieval quality and add warning if needed
        retrieval_warning = ""
        if len(similar_examples) == 0:
            retrieval_warning = "⚠️ **No Training Examples Found**: No similar tracks found in database. Feedback will be very general.\n\n"
        elif len(similar_examples) < 3:
            retrieval_warning = f"⚠️ **Limited Training Data**: Only found {len(similar_examples)} similar examples. Feedback quality may be limited.\n\n"

        # Add debug info (commented out for cleaner demo UI)
        # if similar_examples:
        #     total_feedback_items = sum(len(ex.get("feedback", [])) for ex in similar_examples)
        #     retrieval_warning += f"📊 **Debug Info**: Retrieved {len(similar_examples)} examples with {total_feedback_items} feedback items\n\n"

        # Format examples for prompt
        formatted_examples = self.format_examples_for_prompt(
            similar_examples, user_upload, question
        )

        # Get input and reference track data for the prompt
        session = self.db.get_session()
        try:
            input_track_id = getattr(user_upload, "input_track_id", None)
            reference_track_id = getattr(user_upload, "reference_track_id", None)

            input_track = (
                session.query(Track).filter(Track.id == input_track_id).first()
                if input_track_id
                else None
            )
            reference_track = (
                session.query(Track).filter(Track.id == reference_track_id).first()
                if reference_track_id
                else None
            )
            input_pattern = "Unknown"
            reference_pattern = "Unknown"
            input_raw_pattern = "Unknown"
            reference_raw_pattern = "Unknown"

            if input_track and hasattr(input_track, "smoothed_arrangement_pattern"):
                pattern = getattr(input_track, "smoothed_arrangement_pattern")
                input_pattern = (
                    pattern
                    if pattern and not hasattr(pattern, "__table__")
                    else "Unknown"
                )

            if reference_track and hasattr(
                reference_track, "smoothed_arrangement_pattern"
            ):
                pattern = getattr(reference_track, "smoothed_arrangement_pattern")
                reference_pattern = (
                    pattern
                    if pattern and not hasattr(pattern, "__table__")
                    else "Unknown"
                )

            if input_track and hasattr(input_track, "raw_arrangement_pattern"):
                raw_pattern = getattr(input_track, "raw_arrangement_pattern")
                input_raw_pattern = (
                    raw_pattern
                    if raw_pattern and not hasattr(raw_pattern, "__table__")
                    else "Unknown"
                )

            if reference_track and hasattr(reference_track, "raw_arrangement_pattern"):
                raw_pattern = getattr(reference_track, "raw_arrangement_pattern")
                reference_raw_pattern = (
                    raw_pattern
                    if raw_pattern and not hasattr(raw_pattern, "__table__")
                    else "Unknown"
                )

            # Get global features for comparison
            input_features = {}
            ref_features = {}

            if input_track and hasattr(input_track, "global_feature_data"):
                features = getattr(input_track, "global_feature_data")
                input_features = (
                    features if features and not hasattr(features, "__table__") else {}
                )

            if reference_track and hasattr(reference_track, "global_feature_data"):
                features = getattr(reference_track, "global_feature_data")
                ref_features = (
                    features if features and not hasattr(features, "__table__") else {}
                )

            # Create feature comparison
            feature_comparison = FeatureComparisonService.create_feature_comparison(
                input_features, ref_features
            )
        finally:
            session.close()

        # Prepare input for the chain
        chain_input = {
            "examples": formatted_examples,
            "question": (
                question
                if question
                else f"Please provide feedback on my {getattr(user_upload, 'genre', 'unknown')} track."
            ),
            "input_pattern": input_pattern,
            "input_raw_pattern": input_raw_pattern,
            "reference_pattern": reference_pattern,
            "reference_raw_pattern": reference_raw_pattern,
            "input_features": (
                str(input_features) if input_features else "No features available"
            ),
            "ref_features": (
                str(ref_features) if ref_features else "No features available"
            ),
            "feature_comparison": feature_comparison,
            "genre": getattr(user_upload, "genre", "unknown"),
            "stage": getattr(user_upload, "stage", "unknown"),
        }

        rag_text_formatter = RagTextFormatter(self.operations)

        # Generate feedback using the pre-initialized RAG chain
        try:
            feedback = self.chain.invoke(chain_input)
            # Clean the feedback to remove <think> tags and prepend retrieval warning
            cleaned_feedback = rag_text_formatter.clean_llm_output(feedback)
            return retrieval_warning + cleaned_feedback
        except Exception as e:
            print(f"Error generating feedback with LLM: {e}")
            # Fallback to returning formatted prompt if LLM fails
            fallback_feedback = self.prompt.format(**chain_input)
            cleaned_fallback = rag_text_formatter.clean_llm_output(fallback_feedback)
            return retrieval_warning + cleaned_fallback

    def _build_training_example_results(
        self, similar_tracks: List[Track], k: int, session
    ) -> List[Dict[str, Any]]:
        results = []
        for track in similar_tracks:
            if len(results) >= k:
                break

            # Find training examples for this track
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
                        "embedding": list(track.global_embedding.tolist()),
                        "duration": track.duration,
                        "sample_rate": track.sample_rate,
                        "arrangement_pattern": track.smoothed_arrangement_pattern,
                    },
                    "reference_track": (
                        {
                            "id": reference_track.id,
                            "file_path": reference_track.file_path,
                            "embedding": (
                                list(reference_track.global_embedding.tolist())
                                if reference_track.global_embedding is not None
                                else None
                            ),
                            "duration": reference_track.duration,
                            "sample_rate": reference_track.sample_rate,
                            "arrangement_pattern": reference_track.smoothed_arrangement_pattern,
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

        return results

    def _build_user_context(self, user_upload: UserUpload) -> str:
        """Build user context section with track arrangement patterns"""
        # Get input and reference track arrangement patterns
        session = self.db.get_session()
        try:
            input_track = (
                session.query(Track)
                .filter(Track.id == user_upload.input_track_id)
                .first()
            )
            reference_track = (
                session.query(Track)
                .filter(Track.id == user_upload.reference_track_id)
                .first()
            )
            input_arrangement = (
                input_track.smoothed_arrangement_pattern if input_track else "Unknown"
            )
            reference_arrangement = (
                reference_track.smoothed_arrangement_pattern
                if reference_track
                else "Unknown"
            )
        finally:
            session.close()

        # Build context string
        context = f"User Upload Context:\n"
        context += f"  User Prompt Notes: {user_upload.user_prompt}\n"
        context += f"  Stage: {user_upload.stage}\n"
        context += f"  Genre: {user_upload.genre}\n"
        context += f"  Input Track Structure: {input_arrangement}\n"
        context += f"  Reference Track Structure: {reference_arrangement}\n\n"

        return context

    def _rank_and_select_feedback(
        self,
        similar_examples: List[Dict[str, Any]],
        user_question: str,
        user_genre: str,
    ) -> List[Dict]:
        """Collect and rank feedback from similar examples"""
        # Collect ALL feedback pieces from all similar examples for ranking
        all_feedback = []
        for example in similar_examples:
            feedback_items = example.get("feedback", [])
            for feedback in feedback_items:
                # Add context about which example this feedback came from
                feedback_with_context = feedback.copy()
                feedback_with_context["source_example"] = example
                all_feedback.append(feedback_with_context)

        # Use the formatter for ranking
        formatter = RagTextFormatter(self.operations)
        return formatter.rank_feedback_by_relevance(
            all_feedback, user_question, user_genre
        )

    def _check_ollama_connection(self) -> bool:
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


# Example usage to test in development
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
        similar_examples, user_upload, retrieval_summary = (
            rag.retrieve_similar_examples(user_upload_id=1, k=3)
        )

        # Use the formatter
        formatter = RagTextFormatter(rag.operations)
        formatted_examples = formatter.format_examples_for_prompt(
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

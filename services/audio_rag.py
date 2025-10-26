from typing import List, Dict, Any, Tuple
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


def create_llm_chain(prompts: Dict[str, Any], llm_model: str = "qwen3:8b"):
    """Helper function to create LLM chain with default settings"""
    # Create prompt template
    template = prompts.get("feedback_generation", {}).get("template", "")
    if not template:
        print("Warning: Could not load feedback_generation template from prompts")
        template = "You are an AI music mentor. Provide feedback on: {question}"
    prompt = ChatPromptTemplate.from_template(template)

    # Create LLM
    base_url = os.environ["DEVELOPMENT_BASE_URL"]
    llm = ChatOllama(model=llm_model, temperature=0.4, base_url=base_url)

    # Create chain
    output_parser = StrOutputParser()
    return prompt | llm | output_parser


class AudioRAG:
    def __init__(
        self, operations: AudioRAGOperations, prompts: Dict[str, Any], llm_chain: Any
    ):
        self.operations = operations
        self.db = operations.db
        self.prompts = prompts
        self.chain = llm_chain

        os.environ["LANGCHAIN_TRACING_V2"] = "true"

        # Initialize text formatter for output processing
        self.text_formatter = RagTextFormatter(self.operations)

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

            # This will be captured in the trace output in langsmith
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
                getattr(user_upload, "user_prompt", "general feedback")
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

    @traceable
    def generate_feedback(
        self, user_upload_id: int, question: str = "", k: int = 5
    ) -> str:
        """
        Complete RAG pipeline: retrieve, format, prompt, and generate feedback
        """
        # Retrieve and format
        similar_examples, user_upload, retrieval_info = self.retrieve_similar_examples(
            user_upload_id, k=k
        )
        retrieval_warning = self._generate_retrieval_warnings(similar_examples)
        formatted_examples = self.format_examples_for_prompt(
            similar_examples, user_upload, question
        )

        # Build input and execute
        chain_input = self._build_chain_input(user_upload, formatted_examples, question)
        return self._execute_llm_chain(chain_input, retrieval_warning)

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

    def _generate_retrieval_warnings(
        self, similar_examples: List[Dict[str, Any]]
    ) -> str:
        """Generate warning messages based on retrieval quality"""
        if len(similar_examples) == 0:
            return "⚠️ **No Training Examples Found**: No similar tracks found in database. Feedback will be very general.\n\n"
        elif len(similar_examples) < 3:
            return f"⚠️ **Limited Training Data**: Only found {len(similar_examples)} similar examples. Feedback quality may be limited.\n\n"
        return ""

    def _build_chain_input(
        self, user_upload: UserUpload, formatted_examples: str, question: str
    ) -> Dict[str, str]:
        """Build input dictionary for the LLM chain"""
        # Extract track features and patterns
        track_data = self._extract_track_features(user_upload)

        return {
            "examples": formatted_examples,
            "question": (
                question
                if question
                else f"Please provide feedback on my {getattr(user_upload, 'genre', 'unknown')} track."
            ),
            "input_pattern": track_data["input_pattern"],
            "input_raw_pattern": track_data["input_raw_pattern"],
            "reference_pattern": track_data["reference_pattern"],
            "reference_raw_pattern": track_data["reference_raw_pattern"],
            "input_features": (
                str(track_data["input_features"])
                if track_data["input_features"]
                else "No features available"
            ),
            "ref_features": (
                str(track_data["ref_features"])
                if track_data["ref_features"]
                else "No features available"
            ),
            "feature_comparison": track_data["feature_comparison"],
            "genre": getattr(user_upload, "genre", "unknown"),
            "stage": getattr(user_upload, "stage", "unknown"),
        }

    def _extract_track_features(self, user_upload: UserUpload) -> Dict[str, Any]:
        """Extract track data and features for the prompt"""
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

            # Extract arrangement patterns
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

            return {
                "input_pattern": input_pattern,
                "reference_pattern": reference_pattern,
                "input_raw_pattern": input_raw_pattern,
                "reference_raw_pattern": reference_raw_pattern,
                "input_features": input_features,
                "ref_features": ref_features,
                "feature_comparison": feature_comparison,
            }
        finally:
            session.close()

    def _execute_llm_chain(
        self, chain_input: Dict[str, str], retrieval_warning: str
    ) -> str:
        """Execute the LLM chain and return cleaned output"""
        try:
            feedback = self.chain.invoke(chain_input)
            cleaned_feedback = self.text_formatter.clean_llm_output(feedback)
            return retrieval_warning + cleaned_feedback
        except Exception as e:
            error_msg = f"⚠️ **AI Service Error**: Unable to generate feedback ({str(e)[:100]}...)\n\n"

            # Return a helpful error message with the context that was gathered
            fallback_response = (
                "I was able to analyze your tracks and find similar examples, "
                "but the AI feedback service is currently unavailable. "
                "Please try again in a moment.\n\n"
                "**Track Analysis Summary:**\n"
                f"- Input pattern: {chain_input.get('input_pattern', 'Unknown')}\n"
                f"- Reference pattern: {chain_input.get('reference_pattern', 'Unknown')}\n"
                f"- Genre: {chain_input.get('genre', 'Unknown')}\n"
                f"- Stage: {chain_input.get('stage', 'Unknown')}\n\n"
                "The system found relevant examples for your track type. "
                "Once the AI service is restored, you'll get detailed feedback based on these matches."
            )

            return error_msg + fallback_response

    def _check_ollama_connection(self) -> bool:
        """
        Check if Ollama server is running and accessible
        """
        try:
            # Simple test to see if we can reach the LLM through the chain
            self.chain.invoke({"question": "Hello"})
            return True
        except Exception as e:
            print(f"Ollama connection failed: {e}")
            print("Make sure Ollama is running with: ollama serve")
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
    operations = AudioRAGOperations(db)
    prompts = PromptLoader._load_prompts()
    llm_chain = create_llm_chain(prompts)
    rag = AudioRAG(operations, prompts, llm_chain)

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

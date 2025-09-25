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
            model=llm_model, temperature=0.5, base_url="http://localhost:11434"
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

            # Use the new optimized method that only searches tracks with training examples
            similar_tracks = self.operations.find_similar_tracks_with_training_examples(
                embedding=list(input_track.global_embedding),
                metric=metric,
                limit=k  # No need to multiply since we're pre-filtering
            )

            print(f"DEBUG: Found {len(similar_tracks)} similar tracks with training examples")

            if not similar_tracks:
                print("DEBUG: No similar tracks with training examples found")
                return [], user_upload, {
                    "user_upload_id": user_upload_id,
                    "k_requested": k,
                    "k_found": 0,
                    "metric": metric,
                    "user_genre": user_upload.genre if user_upload else None,
                    "retrieved_tracks": [],
                }

            # Build results from similar tracks (which already have training examples)
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
                            "embedding": list(track.global_embedding),
                            "duration": track.duration,
                            "sample_rate": track.sample_rate,
                            "arrangement_pattern": track.smoothed_arrangement_pattern,
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

    def create_feature_comparison(self, input_features: Dict, ref_features: Dict) -> str:
        """
        Create a comparative analysis between input and reference track features
        """
        if not input_features or not ref_features:
            return "Feature comparison not available - missing feature data."
        
        comparison = "## Analysis Against Your Reference Track:\n\n"
        
        # Helper function to describe differences
        def describe_diff(input_val, ref_val, metric_name, unit=""):
            if input_val is None or ref_val is None:
                return f"  - {metric_name}: Data not available\n"
            
            diff = input_val - ref_val
            diff_pct = (diff / ref_val * 100) if ref_val != 0 else 0
            
            if abs(diff_pct) < 10:  # Less than 10% difference
                return f"  - {metric_name}: Very similar ({input_val:.2f}{unit} vs {ref_val:.2f}{unit})\n"
            elif diff > 0:
                magnitude = "significantly" if abs(diff_pct) > 30 else "moderately"
                return f"  - {metric_name}: Your track is {magnitude} higher ({input_val:.2f}{unit} vs {ref_val:.2f}{unit}, +{diff_pct:.0f}%)\n"
            else:
                magnitude = "significantly" if abs(diff_pct) > 30 else "moderately"
                return f"  - {metric_name}: Your track is {magnitude} lower ({input_val:.2f}{unit} vs {ref_val:.2f}{unit}, {diff_pct:.0f}%)\n"
        
        # Rhythm comparison
        rhythm_input = input_features.get("rhythm", {})
        rhythm_ref = ref_features.get("rhythm", {})
        
        if rhythm_input and rhythm_ref:
            comparison += "**Rhythmic Character:**\n"
            comparison += describe_diff(rhythm_input.get("tempo"), rhythm_ref.get("tempo"), "Tempo", " BPM")
            comparison += describe_diff(rhythm_input.get("onset_density"), rhythm_ref.get("onset_density"), "Rhythmic Activity", " events/sec")
            comparison += describe_diff(rhythm_input.get("beat_strength"), rhythm_ref.get("beat_strength"), "Beat Presence", "")
            comparison += "\n"
        
        # Energy comparison  
        energy_input = input_features.get("energy", {})
        energy_ref = ref_features.get("energy", {})
        
        if energy_input and energy_ref:
            comparison += "**Energy Profile:**\n"
            comparison += describe_diff(energy_input.get("dynamic_range"), energy_ref.get("dynamic_range"), "Dynamic Range", "")
            comparison += describe_diff(energy_input.get("average_energy"), energy_ref.get("average_energy"), "Overall Intensity", "")
            comparison += describe_diff(energy_input.get("peak_density"), energy_ref.get("peak_density"), "Energy Peaks", " /sec")
            comparison += "\n"
        
        # Frequency/EQ comparison
        freq_input = input_features.get("frequency", {})
        freq_ref = ref_features.get("frequency", {})
        
        if freq_input and freq_ref:
            comparison += "**Frequency Distribution:**\n"
            comparison += describe_diff(freq_input.get("low_proportion"), freq_ref.get("low_proportion"), "Bass Content", "%")
            comparison += describe_diff(freq_input.get("mid_proportion"), freq_ref.get("mid_proportion"), "Midrange Content", "%")
            comparison += describe_diff(freq_input.get("high_proportion"), freq_ref.get("high_proportion"), "Treble Content", "%")
            comparison += "\n"
        
        # Spectral comparison
        spectral_input = input_features.get("spectral", {})
        spectral_ref = ref_features.get("spectral", {})
        
        if spectral_input and spectral_ref:
            comparison += "**Tonal Character:**\n"
            comparison += describe_diff(spectral_input.get("avg_brightness"), spectral_ref.get("avg_brightness"), "Overall Brightness", " Hz")
            comparison += "\n"
        
        comparison += "This analysis describes the measurable differences between your input track and reference track to provide context for the feedback below.\n\n"
        
        return comparison

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

        # Get input and reference track arrangement patterns
        session = self.db.get_session()
        try:
            input_track = session.query(Track).filter(Track.id == user_upload.input_track_id).first()
            reference_track = session.query(Track).filter(Track.id == user_upload.reference_track_id).first()
            input_arrangement = input_track.smoothed_arrangement_pattern if input_track else "Unknown"
            reference_arrangement = reference_track.smoothed_arrangement_pattern if reference_track else "Unknown"
        finally:
            session.close()

        # Start with user context
        context = f"User Upload Context:\n"
        context += f"  User Prompt Notes: {user_upload.user_prompt}\n"
        context += f"  Stage: {user_upload.stage}\n"
        context += f"  Genre: {user_upload.genre}\n"
        context += f"  Input Track Structure: {input_arrangement}\n"
        context += f"  Reference Track Structure: {reference_arrangement}\n\n"
        context += "Similar Examples:\n\n"

        formatted_examples = []

        for i, example in enumerate(similar_examples, 1):
            example_text = f"Example {i}:"

            # Add basic example track info
            example_track = example.get("example_track", {})
            reference_track = example.get("reference_track", {})
            example_text += f"\n  Input Track: {os.path.basename(example_track.get('file_path', 'Unknown'))}"
            example_text += f"\n  Input Structure: {example_track.get('arrangement_pattern', 'Unknown')}"
            if reference_track:
                example_text += f"\n  Reference Structure: {reference_track.get('arrangement_pattern', 'Unknown')}"
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

        ARRANGEMENT STRUCTURE NOTATION:
        The tracks use this structure pattern notation:
        - O = Outro/Intro sections (often striped down loops that are simple)
        - A = Verse/Medium energy groove sections (core groove content, they usually lead together a B section or help set the drum groove for the track)
        - B = Chorus/High energy sections (high energy, memorable parts often with more mid frequency content)
        - C = Breakdown sections (transitional or stripped-down ambients parts with no or little drum content)

        CRITICAL: PAY ATTENTION TO THE USER'S CURRENT TRACK STRUCTURE
        The user's track currently has this arrangement pattern: {input_pattern}
        The reference track they're aiming for has: {reference_pattern}

        ARRANGEMENT FEEDBACK RULES:
        1. ONLY suggest improvements based on sections that currently exist in their track
        2. DO NOT suggest adding sections that already exist in the user's pattern: {input_pattern}
        3. DO NOT suggest section sequences that already exist (e.g., if pattern has "A-B", don't suggest "add B after A")
        4. When suggesting new sections, be explicit: "You could add a higher energy A section after your current O loop" 
        5. Reference the user's current pattern specifically: "Your track is currently just an O section, which works as a foundation..."
        6. When mentioning sections from examples or reference tracks, always clarify: "Like in your reference track's B section" or "The A section in this example shows..."
        7. NEVER assume sections exist that aren't in the input pattern
        8. Check the pattern {input_pattern} before making any arrangement suggestions

        You have access to training examples showing how other tracks were improved. Each example contains:
        - An INPUT TRACK (unfinished, like the user's track)
        - A REFERENCE TRACK (finished version)  
        - FEEDBACK explaining how to get from input to reference in the tone of one specific producer.

        {feature_comparison}

        Here are the relevant training examples:

        {examples}

        CRITICAL PATTERN RULES - FOLLOW EXACTLY:
        - User's current track pattern: {input_pattern}
        - Reference track pattern: {reference_pattern}
        - ONLY reference sections that actually exist in these patterns
        - DO NOT mention sections (A, B, C, O) that are not in the user's pattern
        - DO NOT invent arrangement details not provided in the training examples
        - If the user has pattern "O", only discuss the O section and potential additions
        - If suggesting new sections, be explicit: "You could add an A section after your O loop"

        CRITICAL INSTRUCTIONS:
        - The user has uploaded a {stage} track 
        - Their current track structure is: {input_pattern}
        - You do NOT know what specific elements are already in the user's track beyond its arrangement structure
        - Use ONLY the feedback patterns from the training examples - do not add generic music production advice
        - Frame suggestions as "you could try..." or "consider adding..." rather than "your track has..." or "change your..."
        - Base all advice on the provided training examples, not general music knowledge

        User's Question: {question}
        User's Track Context: {stage} track with pattern {input_pattern} needing arrangement help

        Start your response with a descriptive statement about the user's input track. Use the feature comparison data above and the genre information to create 2-3 sentences describing the track's current character, style, and sonic profile. This sets the context for your feedback.

        Then provide targeted feedback. Analyze the user's question to determine what type of feedback they need most:
        - If they ask about structure, arrangement, or song flow → focus on ARRANGEMENT
        - If they ask about drums, rhythm, or groove → focus on RHYTHM  
        - If they ask about energy, power, or dynamics → focus on ENERGY
        - If they ask about EQ, mix, or sound → focus on MIXING/EQ
        - If their question is general → provide 1-2 most relevant categories

        Provide targeted feedback based on their specific question, keeping in mind the feature differences between the input and reference tracks. Use clear headers for the relevant categories:

        **[RELEVANT CATEGORY]:**
        Based on the user's current {input_pattern} structure and similar examples, provide specific advice addressing their question. Consider the measurable differences noted above when making recommendations...

        Remember: The user's track currently has pattern {input_pattern}. Always reference this when giving advice and be clear about what sections exist vs. what could be added. Focus on answering their specific question while incorporating insights from the feature analysis.

        IMPORTANT: Your response should ONLY contain the music feedback for the user. Do not include any of these instructions or meta-commentary in your output.

        ---
        RESPONSE:"""

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

        # Check retrieval quality and add warning if needed
        retrieval_warning = ""
        if len(similar_examples) == 0:
            retrieval_warning = "⚠️ **No Training Examples Found**: No similar tracks found in database. Feedback will be very general.\n\n"
        elif len(similar_examples) < 3:
            retrieval_warning = f"⚠️ **Limited Training Data**: Only found {len(similar_examples)} similar examples. Feedback quality may be limited.\n\n"
        
        # Add debug info
        if similar_examples:
            total_feedback_items = sum(len(ex.get("feedback", [])) for ex in similar_examples)
            retrieval_warning += f"📊 **Debug Info**: Retrieved {len(similar_examples)} examples with {total_feedback_items} feedback items\n\n"

        # Format examples for prompt
        formatted_examples = self.format_examples_for_prompt(
            similar_examples, user_upload
        )

        # Get input and reference track data for the prompt
        session = self.db.get_session()
        try:
            input_track = session.query(Track).filter(Track.id == user_upload.input_track_id).first()
            reference_track = session.query(Track).filter(Track.id == user_upload.reference_track_id).first()
            input_pattern = input_track.smoothed_arrangement_pattern if input_track and hasattr(input_track, 'smoothed_arrangement_pattern') and input_track.smoothed_arrangement_pattern else "Unknown"
            reference_pattern = reference_track.smoothed_arrangement_pattern if reference_track and hasattr(reference_track, 'smoothed_arrangement_pattern') and reference_track.smoothed_arrangement_pattern else "Unknown"
            
            # Get global features for comparison
            input_features = input_track.global_feature_data if input_track and input_track.global_feature_data else {}
            ref_features = reference_track.global_feature_data if reference_track and reference_track.global_feature_data else {}
            
            print(f"DEBUG: Input features available: {bool(input_features)}")
            print(f"DEBUG: Ref features available: {bool(ref_features)}")
            if input_features:
                print(f"DEBUG: Input feature keys: {list(input_features.keys())}")
            if ref_features:
                print(f"DEBUG: Ref feature keys: {list(ref_features.keys())}")
            
            # Create feature comparison
            feature_comparison = self.create_feature_comparison(input_features, ref_features)
            print(f"DEBUG: Feature comparison length: {len(feature_comparison)} chars")
        finally:
            session.close()

        # Prepare input for the chain
        chain_input = {
            "examples": formatted_examples,
            "question": (
                question
                if question
                else f"Please provide feedback on my {user_upload.genre} track."
            ),
            "input_pattern": input_pattern,
            "reference_pattern": reference_pattern,
            "feature_comparison": feature_comparison,
            "genre": user_upload.genre,
            "stage": user_upload.stage,
        }

        # Generate feedback using the pre-initialized RAG chain
        try:
            feedback = self.chain.invoke(chain_input)
            # Prepend retrieval warning to feedback
            return retrieval_warning + feedback
        except Exception as e:
            print(f"Error generating feedback with LLM: {e}")
            # Fallback to returning formatted prompt if LLM fails
            return retrieval_warning + self.prompt.format(**chain_input)


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

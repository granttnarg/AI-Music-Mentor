from typing import List, Dict, Any, Tuple
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
    def __init__(self, db: AudioRAGDatabase, llm_model: str = "qwen3:8b"):
        self.db = db
        self.operations = AudioRAGOperations(db)
        self.llm_model = llm_model

        os.environ["LANGCHAIN_TRACING_V2"] = "true"

        # Initialize RAG components
        self.prompt = self.create_prompt_template()
        self.output_parser = StrOutputParser()
        self.llm = ChatOllama(
            model=llm_model, temperature=0.4, base_url="http://localhost:11434"
        )
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

    def rank_feedback_by_relevance(self, feedback_items: List[Dict], user_question: str, user_genre: str) -> List[Dict]:
        """
        Use a small LLM to rank feedback items by relevance to user question
        Returns top-ranked feedback pieces
        """
        if not feedback_items or len(feedback_items) <= 2:
            return feedback_items  # If we have 2 or fewer, use all
            
        try:
            # Initialize small, fast model for ranking
            ranking_llm = ChatOllama(
                model="llama3.2:latest",  # Use available model for ranking
                temperature=0.0,  # Zero temperature for consistent scoring
            )
            
            scored_feedback = []
            
            for feedback in feedback_items:
                feedback_text = feedback.get('text', '')
                feedback_type = feedback.get('type', 'general')
                
                # Create ranking prompt
                ranking_prompt = f"""Rate how relevant this feedback is to the user's question. Use the full 1-10 scale:

                User Question: "{user_question}"
                User Genre: {user_genre} 
                Feedback Type: {feedback_type}
                Feedback Text: "{feedback_text}"

                Scoring guidelines:
                - 1-3: Completely unrelated to the question (wrong topic entirely)
                - 4-6: Somewhat related but doesn't directly address the question  
                - 7-9: Directly addresses the question and would help solve the problem
                - 10: Perfect match - exactly what the user needs

                Be decisive. Don't default to middle scores. If this feedback doesn't directly solve their specific problem, score it low.

                Score (1-10):"""

                try:
                    print(f"\nDEBUG: Scoring feedback: '{feedback_text[:60]}...'")
                    print(f"DEBUG: Feedback type: {feedback_type}")
                    print(f"DEBUG: User question: '{user_question[:60]}...'")
                    
                    # Get relevance score from small LLM
                    response = ranking_llm.invoke(ranking_prompt)
                    score_text = response.content.strip()
                    print(f"DEBUG: Raw LLM response: '{score_text}'")
                    
                    # Extract numeric score
                    import re
                    score_match = re.search(r'\b([1-9]|10)\b', score_text)
                    score = int(score_match.group(1)) if score_match else 5  # Default to 5 if parsing fails
                    
                    print(f"DEBUG: Extracted score: {score}")
                    
                    scored_feedback.append({
                        'feedback': feedback,
                        'score': score
                    })
                    
                except Exception as e:
                    print(f"ERROR: Failed to score feedback: {e}")
                    print(f"DEBUG: Feedback text was: '{feedback_text}'")
                    # Fallback: give average score
                    scored_feedback.append({
                        'feedback': feedback,
                        'score': 5
                    })
            
            # Sort by score and return top pieces
            scored_feedback.sort(key=lambda x: x['score'], reverse=True)
            top_feedback = [item['feedback'] for item in scored_feedback[:2]]  # Top 2 most relevant
            
            print(f"\n🎯 RANKING RESULTS: Scored {len(feedback_items)} feedback pieces, selected top {len(top_feedback)}")
            print("=" * 80)
            for i, item in enumerate(scored_feedback):  # Show ALL scores for debugging
                feedback_text = item['feedback'].get('text', 'No text')[:70]
                feedback_type = item['feedback'].get('type', 'General')
                selected = "✅ SELECTED" if i < 2 else "❌ rejected"
                print(f"#{i+1:2d} | Score: {item['score']:2d} | {selected} | Type: {feedback_type}")
                print(f"     | Text: {feedback_text}...")
                print(f"     |")
            print("=" * 80)
                
            return top_feedback
            
        except Exception as e:
            print(f"ERROR: Feedback ranking failed: {e}")
            # Fallback to first 2 feedback pieces
            return feedback_items[:2]

    @traceable
    def format_examples_for_prompt(
        self, similar_examples: List[Dict[str, Any]], user_upload: UserUpload, question: str = ""
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
        # Collect ALL feedback pieces from all similar examples for ranking
        all_feedback = []
        for example in similar_examples:
            feedback_items = example.get("feedback", [])
            for feedback in feedback_items:
                # Add context about which example this feedback came from
                feedback_with_context = feedback.copy()
                feedback_with_context['source_example'] = example
                all_feedback.append(feedback_with_context)

        # Rank feedback by relevance using small LLM
        user_question = question if question.strip() else (getattr(user_upload, 'user_prompt_notes', 'general feedback') or "general feedback")
        user_genre = getattr(user_upload, 'genre', 'electronic')
        
        print(f"DEBUG: Ranking {len(all_feedback)} total feedback pieces...")
        ranked_feedback = self.rank_feedback_by_relevance(all_feedback, user_question, user_genre)
        
        context += f"Most Relevant Feedback (from {len(similar_examples)} similar tracks):\n\n"

        formatted_examples = []
        # Format only the top-ranked feedback pieces
        for i, feedback in enumerate(ranked_feedback, 1):
            source_example = feedback.get('source_example', {})
            example_track = source_example.get("example_track", {})
            
            feedback_text = f"Relevant Example {i}:"
            feedback_text += f"\n  Source Track: {os.path.basename(example_track.get('file_path', 'Unknown'))}"
            feedback_text += f"\n  Structure: {example_track.get('arrangement_pattern', 'Unknown')}"
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
        template = """You are an AI music mentor and professional music producer. Your tone is supportive, casual, humourous and constructive.

                Primary goal: Help the user overcome writer's block by providing exactly 2 concrete, actionable suggestions that move their track forward. You are not trying to complete their track or provide a comprehensive roadmap - just give them the 2 most impactful, specific ideas to break through their current sticking point and continue creating. Keep feedback focused and digestible.

                The reference track serves as a directional guide, NOT a rigid template. The goal is to move the input track in that musical direction while respecting the track's current state and ensuring all changes make musical sense for where the track currently is.

                Track Section Definitions - USE THESE EXACT TERMS:
                - O = Intro/Outro sections: simple loops, often stripped-down rhythmic loops, setting the foundation for DJ mixing
                - A = Groove sections: medium energy core groove content with steady rhythmic patterns, usually leading into B sections and keeping the track moving forward
                - B = Main Hook sections: high-energy memorable groove sections with prominent melodies, catchy rhythmic motifs, or increased mid-frequency content
                - C = Breakdown sections: transitional or ambient parts with minimal or no drums, often used for emotional peaks or breaks from the main groove

                PATTERN INTERPRETATION:
                - If input track pattern is "O": Track contains ONLY intro/outro material. No A, B, or C sections exist yet.
                - If input track pattern is "A": Track contains ONLY groove material. No B or C sections exist yet.
                - If input track pattern is "O-A": Track has intro/outro material followed by groove material. No B or C sections exist yet.
                - Only suggest adding sections that don't already exist, never extending or modifying sections that aren't there.

                CRITICAL: Always use these exact section names
                - Use full section names with optional letter clarification: "Groove sections (A)", "Main Hook sections (B)", etc.
                - Acceptable: "your 2nd Groove section", "the first Main Hook section (B)", "Breakdown sections (C)", "trim the last Main Hook section by 4 chunks"
                - Not acceptable: "B sections", "A sections", "the B", "your A section"

                Arrangement Guidance:
                Always use the reference track's structure as your primary guide when analyzing or suggesting an arrangement.

                You may also reference the common techno track patterns below, as they represent popular and functional choices for engaging the dancefloor.

                When referencing these patterns, explain why they work (e.g., how tension builds, how breakdowns add contrast, how O sections aid DJ mixing).

                Techno Tracks typically start and end with an O or A section: a stripped groove that DJs can easily mix in and out.

                B sections work best after A (gradual build into a drop) or C (impactful re-entry after a breakdown).

                Common Techno Track Patterns:
                - O–A–B–C–B–A–O → Builds to high energy, drops into a memorable breakdown, then re-energizes before winding down. "Orbital – Chime" is a great example of this, ending with the chorus and short outro.
                - O–A–B–C–A–B–O → Builds to high energy, uses an emotive breakdown, recalls the chorus energy near the end. "Laurent Garnier – The Man With The Red Face" a perfect example of this big build emotive breakdown gradually building back to the chorus.
                - O–A–A–B–B–A–O → Gradual climb to a strong mid-track peak, then a taper down; best with hypnotic grooves where breakdowns may interrupt flow. - "Robert Hood - Rhythm of vision" is good example of this style of building track.
                - O–A–B–A–B–A–B–O → Alternates between medium and high energy, playing with tension over a long, dynamic journey. "Jeff Mills - The Bells" is a good example of this structure.

                If it makes sense for your arrangement advice, you may mention only 1 of the classic techno tracks by artist and name from the common techno patterns above, suggesting the user checks out that track for more inspiration.

                Input:
                - User track stage/state: {stage} (sketch, half-finished, almost finished)
                - Genre: {genre}
                - Input track features: {input_features} (use this to describe the track's current characteristics and identify potential problems)
                - Reference track features: {ref_features}
                - Feature comparison analysis: {feature_comparison} (use this to understand specific gaps between input and reference, and guide suggestions for bridging those gaps toward the reference direction, not exact replication)
                - Input track pattern: {input_pattern} - THIS IS YOUR PRIMARY FOCUS
                - Input raw pattern: {input_raw_pattern} (detailed timing - use this ONLY to identify unusually long/short sections, not for suggestions)
                - Reference track pattern: {reference_pattern} (inspiration only - use for creative ideas, not rigid copying)
                - Reference raw pattern: {reference_raw_pattern} (reference timing for creative inspiration only)
                - User question: {question}

                IMPORTANT TIMING AND PATTERN NOTES:
                - Always reference the COMPRESSED pattern of the input track when suggesting changes
                - Use the RAW pattern only to identify timing issues (e.g., "your B section is quite long at 12 chunks/48 bars" or "the final O is very short at 2 chunks/8 bars")
                - When suggesting changes, speak in 4-bar chunks: "add 2 chunks (8 bars) after the A" or "trim the B section by 3 chunks (12 bars)"
                - Never reference raw pattern numbers in suggestions (don't say "change 12B to 8B" - say "trim the B section by 4 chunks")

                IMPORTANT TO TYPE OF FEEDBACK: Stay focused on what the user is asking for. If they ask for mix/EQ help, only give mix advice. If they ask for arrangement help, only give arrangement advice. Don't add other types of feedback unless specifically requested.
                - Feedback examples: {examples}

                Instructions:
                1. Internally, reason using a Graph of Thought structure:
                - Nodes = track sections (O, A, B, C) with attributes:
                    - Energy, groove, eq and arrangement characteristics from features
                    - Differences from reference track
                    - Positive aspects that are already working
                    - Musical reasoning for potential improvements
                - Edges = suggested improvements or actions:
                    - Types: "add section", "extend section", "adjust energy", "modify groove", "mix advice", "suggest classic track for arrangement inspiration"
                    - Only suggest changes consistent with the user's current pattern
                    - Include reasoning for each suggestion based on features, reference track, and relevant feedback examples
                2. Filter feedback examples using feature and pattern similarity:
                - Only use examples relevant to the current input track, you can use the {input_features} and {input_pattern} as a guide here.
                - Extract the communication style, tone, and personality from the feedback examples
                - Adopt the producer's voice: use their vocabulary patterns, phrasing style, and approach to giving feedback
                - Integrate actionable advice from examples into your reasoning
                3. Generate readable, user-facing feedback:
                - Begin with a descriptive, supportive analysis of the user's track, highlighting strengths and differences from the reference
                - Provide section-by-section guidance, referencing only sections present in the input or reference pattern
                - Explain *why* each suggested change is musically helpful, using natural producer language
                - Frame suggestions using the producer's typical phrasing patterns (e.g., their preferred way of suggesting changes)
                - Organize feedback by relevant categories based on the user question (e.g., Arrangement, Energy, Rhythm, Mix)
                - Reference relevant feedback examples if they support your advice
                - When using technical analysis, translate to producer language:
                    "onset_density too high" → "simplify the drum pattern"
                    "spectral_centroid low" → "add some high-end sparkle"
                    "mid-range heavy" → "carve space around 200Hz"

                Critical rules:
                - STRICTLY only reference sections that actually exist in the input track pattern. If input track is "O", it has ONLY intro/outro material and NO other sections exist yet.
                - Do not suggest extending or modifying non-existent sections (e.g., if input is "O", don't mention "extending the B section")
                - Do not invent arrangement details or instruments not present in examples
                - Only use feedback examples that are relevant based on feature and pattern similarity
                - Keep all advice musically grounded and actionable
                - Maintain the producer's authentic voice and tone throughout your response
                - NEVER reference examples by number (e.g., "Example 1", "Example 3") in your output
                - Use clear, accessible language instead of raw technical terms (e.g., "peak_density" → "rhythmic intensity", "spectral_centroid" → "brightness")
                - If suggesting EQ changes, provide specific frequency ranges and techniques
                - Focus on the 2 most impactful changes, not a comprehensive overhaul
                - NEVER use technical analysis terms in output: no "onset density", "beat strength", "spectral centroid"
                - Always translate: "onset density too high" → "drum pattern feels busy", "low beat strength" → "kick needs more punch"

                Output:
                - A readable, structured feedback report for the user, including:
                - A warm casaul greeting complimenting their track mentioning the {stage} and {genre}
                - Descriptive musical analysis of the track
                - Exactly 2 numbered suggestions with clear reasoning
                - Section-specific advice (arrangement, energy, rhythm, mix) - only include sections relevant to the user question: {question}
                - When describing arrangment advice use the full section names with optional letter clarification: "Groove sections (A)","Main Hook sections (B)", etc.
                - Musical reasoning behind suggestions
                - Actionable steps based on relevant feedback examples only if they make sense
                - A 'Pro-tip' relevant to their track
                - End with supportive encouragement under the heading "FINAL THOUGHTS" with a clear call-to-action encouraging the user to work on their track and come back with a new version for more feedback or producer advice.
                -
                """

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

    def clean_llm_output(self, text: str) -> str:
        """
        Remove <think> tags and their content from LLM output
        """
        import re
        # Remove anything between <think> and </think> tags (case insensitive, multiline)
        cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.IGNORECASE | re.DOTALL)
        # Clean up any extra whitespace that might be left
        cleaned = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned)  # Replace multiple newlines with double newlines
        return cleaned.strip()

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
            input_track_id = getattr(user_upload, 'input_track_id', None)
            reference_track_id = getattr(user_upload, 'reference_track_id', None)
            
            input_track = session.query(Track).filter(Track.id == input_track_id).first() if input_track_id else None
            reference_track = session.query(Track).filter(Track.id == reference_track_id).first() if reference_track_id else None
            input_pattern = "Unknown"
            reference_pattern = "Unknown"
            input_raw_pattern = "Unknown"
            reference_raw_pattern = "Unknown"
            
            if input_track and hasattr(input_track, 'smoothed_arrangement_pattern'):
                pattern = getattr(input_track, 'smoothed_arrangement_pattern')
                input_pattern = pattern if pattern and not hasattr(pattern, '__table__') else "Unknown"
            
            if reference_track and hasattr(reference_track, 'smoothed_arrangement_pattern'):
                pattern = getattr(reference_track, 'smoothed_arrangement_pattern')
                reference_pattern = pattern if pattern and not hasattr(pattern, '__table__') else "Unknown"
                
            if input_track and hasattr(input_track, 'raw_arrangement_pattern'):
                raw_pattern = getattr(input_track, 'raw_arrangement_pattern')
                input_raw_pattern = raw_pattern if raw_pattern and not hasattr(raw_pattern, '__table__') else "Unknown"
            
            if reference_track and hasattr(reference_track, 'raw_arrangement_pattern'):
                raw_pattern = getattr(reference_track, 'raw_arrangement_pattern')
                reference_raw_pattern = raw_pattern if raw_pattern and not hasattr(raw_pattern, '__table__') else "Unknown"
            
            # Get global features for comparison
            input_features = {}
            ref_features = {}
            
            if input_track and hasattr(input_track, 'global_feature_data'):
                features = getattr(input_track, 'global_feature_data')
                input_features = features if features and not hasattr(features, '__table__') else {}
            
            if reference_track and hasattr(reference_track, 'global_feature_data'):
                features = getattr(reference_track, 'global_feature_data')
                ref_features = features if features and not hasattr(features, '__table__') else {}
            
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
                else f"Please provide feedback on my {getattr(user_upload, 'genre', 'unknown')} track."
            ),
            "input_pattern": input_pattern,
            "input_raw_pattern": input_raw_pattern,
            "reference_pattern": reference_pattern,
            "reference_raw_pattern": reference_raw_pattern,
            "input_features": str(input_features) if input_features else "No features available",
            "ref_features": str(ref_features) if ref_features else "No features available", 
            "feature_comparison": feature_comparison,
            "genre": getattr(user_upload, 'genre', 'unknown'),
            "stage": getattr(user_upload, 'stage', 'unknown'),
        }
        
        print(f"DEBUG: Chain input keys: {list(chain_input.keys())}")
        for key, value in chain_input.items():
            print(f"DEBUG: {key}: {type(value)} - {str(value)[:100]}...")
        

        # Generate feedback using the pre-initialized RAG chain
        try:
            feedback = self.chain.invoke(chain_input)
            # Clean the feedback to remove <think> tags and prepend retrieval warning
            cleaned_feedback = self.clean_llm_output(feedback)
            return retrieval_warning + cleaned_feedback
        except Exception as e:
            print(f"Error generating feedback with LLM: {e}")
            # Fallback to returning formatted prompt if LLM fails
            fallback_feedback = self.prompt.format(**chain_input)
            cleaned_fallback = self.clean_llm_output(fallback_feedback)
            return retrieval_warning + cleaned_fallback


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

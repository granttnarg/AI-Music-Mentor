from langsmith import traceable
from db.models import Track, UserUpload
from db.operations import AudioRAGOperations
from typing import Dict, List, Any
from langchain_ollama import ChatOllama
from .prompt_loader import PromptLoader
import os


class RagTextFormatter:
    def __init__(self, db_operations: AudioRAGOperations):
        self.db_operations = db_operations
        self.prompts = PromptLoader._load_prompts()

    def rank_feedback_by_relevance(
        self, feedback_items: List[Dict], user_question: str, user_genre: str
    ) -> List[Dict]:
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
                feedback_text = feedback.get("text", "")
                feedback_type = feedback.get("type", "general")

                # Create ranking prompt from YAML
                ranking_template = self.prompts.get("feedback_ranking", {}).get(
                    "template", ""
                )
                if not ranking_template:
                    # Fallback if YAML not available
                    ranking_template = """Rate this feedback (1-10):
                    Question: "{user_question}"
                    Feedback: "{feedback_text}"
                    Score:"""

                ranking_prompt = ranking_template.format(
                    user_question=user_question,
                    user_genre=user_genre,
                    feedback_type=feedback_type,
                    feedback_text=feedback_text,
                )

                try:
                    # Get relevance score from small LLM
                    response = ranking_llm.invoke(ranking_prompt)
                    score_text = response.content.strip()

                    # Extract numeric score
                    import re

                    score_match = re.search(r"\b([1-9]|10)\b", score_text)
                    score = (
                        int(score_match.group(1)) if score_match else 5
                    )  # Default to 5 if parsing fails

                    scored_feedback.append({"feedback": feedback, "score": score})

                except Exception as e:
                    print(f"ERROR: Failed to score feedback: {e}")
                    # Fallback: give average score
                    scored_feedback.append({"feedback": feedback, "score": 5})

            # Sort by score and return top pieces
            scored_feedback.sort(key=lambda x: x["score"], reverse=True)
            top_feedback = [
                item["feedback"] for item in scored_feedback[:2]
            ]  # Top 2 most relevant

            print(
                f"\n🎯 RANKING RESULTS: Scored {len(feedback_items)} feedback pieces, selected top {len(top_feedback)}"
            )
            print("=" * 80)
            for i, item in enumerate(scored_feedback):  # Show ALL scores for debugging
                feedback_text = item["feedback"].get("text", "No text")[:70]
                feedback_type = item["feedback"].get("type", "General")
                selected = "✅ SELECTED" if i < 2 else "❌ rejected"
                print(
                    f"#{i+1:2d} | Score: {item['score']:2d} | {selected} | Type: {feedback_type}"
                )
                print(f"     | Text: {feedback_text}...")
                print(f"     |")
            print("=" * 80)

            return top_feedback

        except Exception as e:
            print(f"ERROR: Feedback ranking failed: {e}")
            # Fallback to first 2 feedback pieces
            return feedback_items[:2]

    def clean_llm_output(self, text: str) -> str:
        """
        Remove <think> tags and their content from LLM output
        """
        import re

        # Remove anything between <think> and </think> tags (case insensitive, multiline)
        cleaned = re.sub(
            r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL
        )
        # Clean up any extra whitespace that might be left
        cleaned = re.sub(
            r"\n\s*\n\s*\n", "\n\n", cleaned
        )  # Replace multiple newlines with double newlines
        return cleaned.strip()

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

        # Get input and reference track arrangement patterns
        session = self.db_operations.db.get_session()
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
                feedback_with_context["source_example"] = example
                all_feedback.append(feedback_with_context)

        # Rank feedback by relevance using small LLM
        user_question = (
            question
            if question.strip()
            else (
                getattr(user_upload, "user_prompt_notes", "general feedback")
                or "general feedback"
            )
        )
        user_genre = getattr(user_upload, "genre", "electronic")

        print(f"🎯 Ranking {len(all_feedback)} total feedback pieces...")
        ranked_feedback = self.rank_feedback_by_relevance(
            all_feedback, user_question, user_genre
        )

        context += (
            f"Most Relevant Feedback (from {len(similar_examples)} similar tracks):\n\n"
        )

        formatted_examples = []
        # Format only the top-ranked feedback pieces
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

from langsmith import traceable
from db.models import Track, UserUpload
from db.operations import AudioRAGOperations
from typing import Dict, List, Any
from langchain_ollama import ChatOllama
from .prompt_loader import PromptLoader
import os
import re


class RAGTextHelper:
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
            ranking_llm = self._create_ranking_llm()
            scored_feedback = self._score_all_feedback(
                feedback_items, user_question, user_genre, ranking_llm
            )
            top_feedback = self._select_top_feedback(scored_feedback)
            self._log_ranking_results(scored_feedback)
            return top_feedback

        except Exception as e:
            print(f"ERROR: Feedback ranking failed: {e}")
            # Fallback to first 2 feedback pieces
            return feedback_items[:2]

    def _create_ranking_llm(self):
        """Create LLM instance for ranking feedback"""
        return ChatOllama(
            model="llama3.2:latest",  # Use available model for ranking
            temperature=0.0,  # Zero temperature for consistent scoring
        )

    def _score_all_feedback(
        self,
        feedback_items: List[Dict],
        user_question: str,
        user_genre: str,
        ranking_llm,
    ) -> List[Dict]:
        """Score all feedback items using the ranking LLM"""
        scored_feedback = []

        for feedback in feedback_items:
            score = self._score_single_feedback(
                feedback, user_question, user_genre, ranking_llm
            )
            scored_feedback.append({"feedback": feedback, "score": score})

        return scored_feedback

    def _score_single_feedback(
        self, feedback: Dict, user_question: str, user_genre: str, ranking_llm
    ) -> int:
        """Score a single feedback item"""
        feedback_text = feedback.get("text", "")
        feedback_type = feedback.get("type", "general")

        ranking_prompt = self._build_ranking_prompt(
            user_question, user_genre, feedback_type, feedback_text
        )

        try:
            response = ranking_llm.invoke(ranking_prompt)
            return self._extract_score_from_response(response.content.strip())
        except Exception as e:
            print(f"ERROR: Failed to score feedback: {e}")
            return 5  # Default score

    def _build_ranking_prompt(
        self,
        user_question: str,
        user_genre: str,
        feedback_type: str,
        feedback_text: str,
    ) -> str:
        """Build the prompt for ranking feedback"""
        ranking_template = self.prompts.get("feedback_ranking", {}).get("template", "")
        if not ranking_template:
            # Fallback if YAML not available
            ranking_template = """Rate this feedback (1-10):
            Question: "{user_question}"
            Feedback: "{feedback_text}"
            Score:"""

        return ranking_template.format(
            user_question=user_question,
            user_genre=user_genre,
            feedback_type=feedback_type,
            feedback_text=feedback_text,
        )

    def _extract_score_from_response(self, score_text: str) -> int:
        """Extract numeric score from LLM response"""
        import re

        score_match = re.search(r"\b([1-9]|10)\b", score_text)
        return int(score_match.group(1)) if score_match else 5

    def _select_top_feedback(
        self, scored_feedback: List[Dict], top_n: int = 2
    ) -> List[Dict]:
        """Select top N feedback items by score"""
        scored_feedback.sort(key=lambda x: x["score"], reverse=True)
        return [item["feedback"] for item in scored_feedback[:top_n]]

    def _log_ranking_results(self, scored_feedback: List[Dict]):
        """Log ranking results for debugging"""
        print(
            f"\n🎯 RANKING RESULTS: Scored {len(scored_feedback)} feedback pieces, selected top 2"
        )
        print("=" * 80)
        for i, item in enumerate(scored_feedback):
            feedback_text = item["feedback"].get("text", "No text")[:70]
            feedback_type = item["feedback"].get("type", "General")
            selected = "✅ SELECTED" if i < 2 else "❌ rejected"
            print(
                f"#{i+1:2d} | Score: {item['score']:2d} | {selected} | Type: {feedback_type}"
            )
            print(f"     | Text: {feedback_text}...")
            print(f"     |")
        print("=" * 80)

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

        context = self._build_user_context(user_upload)
        ranked_feedback = self._collect_and_rank_feedback(
            similar_examples, user_upload, question
        )
        formatted_examples = self._format_feedback_examples(ranked_feedback)
        summary = self._build_summary(similar_examples)

        return context + formatted_examples + summary

    def _build_user_context(self, user_upload: UserUpload) -> str:
        """Build user context section with track arrangement patterns"""
        input_arrangement, reference_arrangement = self._get_track_arrangements(
            user_upload
        )

        context = f"User Upload Context:\n"
        context += f"  User Prompt Notes: {user_upload.user_prompt}\n"
        context += f"  Stage: {user_upload.stage}\n"
        context += f"  Genre: {user_upload.genre}\n"
        context += f"  Input Track Structure: {input_arrangement}\n"
        context += f"  Reference Track Structure: {reference_arrangement}\n\n"

        return context

    def _get_track_arrangements(self, user_upload: UserUpload):
        """Get arrangement patterns for input and reference tracks"""
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
            return input_arrangement, reference_arrangement
        finally:
            session.close()

    def _collect_and_rank_feedback(
        self,
        similar_examples: List[Dict[str, Any]],
        user_upload: UserUpload,
        question: str,
    ) -> List[Dict]:
        """Collect all feedback and rank by relevance"""
        # Collect ALL feedback pieces from all similar examples for ranking
        all_feedback = []
        for example in similar_examples:
            feedback_items = example.get("feedback", [])
            for feedback in feedback_items:
                # Add context about which example this feedback came from
                feedback_with_context = feedback.copy()
                feedback_with_context["source_example"] = example
                all_feedback.append(feedback_with_context)

        # Determine user question for ranking
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
        return self.rank_feedback_by_relevance(all_feedback, user_question, user_genre)

    def _format_feedback_examples(self, ranked_feedback: List[Dict]) -> str:
        """Format the ranked feedback into example text"""
        header = f"Most Relevant Feedback (from similar tracks):\n\n"

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

        return header + "\n\n".join(formatted_examples)

    def _build_summary(self, similar_examples: List[Dict[str, Any]]) -> str:
        """Build summary of retrieval quality"""
        total_feedback_items = sum(
            len(ex.get("feedback", [])) for ex in similar_examples
        )
        return f"\n\n[Retrieved {len(similar_examples)} examples with {total_feedback_items} total feedback items]"

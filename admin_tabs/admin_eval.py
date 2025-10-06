import streamlit as st
import json
import os
import glob
import matplotlib.pyplot as plt
import pandas as pd
import difflib
from datetime import datetime
from pathlib import Path
from admin_tabs.add_new import get_database


def show_admin_eval_tab():
    """Show the Admin Evaluation tab content"""
    st.markdown("#### Audio Feedback Evaluation Metrics")
    st.caption(
        "Systematically evaluate LLM output quality to track improvements over time"
    )

    # Create evaluations directory if it doesn't exist
    eval_dir = Path("evaluations")
    eval_dir.mkdir(exist_ok=True)

    # Load existing evaluations from individual files
    def load_all_evaluations():
        """Load all evaluation files from the evaluations directory"""
        evaluations = []
        eval_files = glob.glob(str(eval_dir / "eval_*.json"))

        for eval_file in sorted(eval_files):
            try:
                with open(eval_file, "r") as f:
                    evaluation = json.load(f)
                    evaluations.append(evaluation)
            except (json.JSONDecodeError, FileNotFoundError) as e:
                st.warning(f"Could not load evaluation file {eval_file}: {e}")
                continue

        return evaluations

    evaluations = load_all_evaluations()

    # Get latest upload for default values
    def get_latest_upload_info():
        try:
            db_ops = get_database()
            session = db_ops.db.get_session()

            # Get the most recent user upload
            from db.models import UserUpload, Track

            latest_upload = (
                session.query(UserUpload).order_by(UserUpload.id.desc()).first()
            )

            if latest_upload:
                # Get track info
                input_track = (
                    session.query(Track)
                    .filter(Track.id == latest_upload.input_track_id)
                    .first()
                )
                ref_track = (
                    session.query(Track)
                    .filter(Track.id == latest_upload.reference_track_id)
                    .first()
                )

                return {
                    "upload_id": latest_upload.id,
                    "input_filename": (
                        Path(input_track.file_path).name if input_track else ""
                    ),
                    "reference_filename": (
                        Path(ref_track.file_path).name if ref_track else ""
                    ),
                    "session_id": latest_upload.session_id,
                }
        except Exception as e:
            print(f"Error getting latest upload: {e}")
            return {
                "upload_id": None,
                "input_filename": "",
                "reference_filename": "",
                "session_id": None,
            }
        finally:
            if "session" in locals():
                session.close()

    # Get defaults from latest upload
    latest_info = get_latest_upload_info()

    # Form for new evaluation
    st.markdown("### New Evaluation")
    if latest_info["upload_id"]:
        st.caption(
            f"💡 Pre-filled with latest upload info (ID: {latest_info['upload_id']}) - edit as needed"
        )

    with st.form("evaluation_form"):
        # Basic info
        col1, col2 = st.columns(2)
        with col1:
            input_filename = st.text_input(
                "Input Track Filename",
                value=latest_info["input_filename"],
                placeholder="e.g., my_techno_track.mp3",
            )
            reference_filename = st.text_input(
                "Reference Track Filename",
                value=latest_info["reference_filename"],
                placeholder="e.g., reference_track.mp3",
            )
        with col2:
            model_version = st.selectbox(  "Model Version", options=["llama3.2:latest", "qwen3:8b"])
            st.write("You selected:", model_version)
            date = st.date_input("Date", value=datetime.now().date())

        # Optional track ID (from database after processing)
        default_track_id = (
            f"upload_{latest_info['upload_id']}" if latest_info["upload_id"] else ""
        )
        track_id = st.text_input(
            "Track ID (optional)",
            value=default_track_id,
            placeholder="e.g., upload_123 - fill after processing if known",
        )

        # LLM Output
        st.markdown("### LLM Output to Evaluate")
        llm_output = st.text_area(
            "Paste the complete LLM feedback response here:",
            height=200,
            placeholder="Paste the full feedback output from the system...",
        )

        # Optional: Full prompt context
        st.markdown("### Prompt Context (Optional)")
        with st.expander(
            "📝 Paste Full RAG Prompt (helpful for tracking prompt changes)"
        ):
            st.caption(
                "This helps correlate prompt modifications with output quality changes"
            )
            full_prompt = st.text_area(
                "Paste the complete prompt template used for this evaluation:",
                height=300,
                placeholder="Optional: Paste the full prompt template to track what instructions were given to the LLM...",
                key="full_prompt",
            )

        st.markdown("### Evaluation Criteria")
        st.markdown("**Rating Scale: 1 (Poor) to 5 (Excellent)**")

        # Evaluation metrics with detailed descriptions
        st.markdown("#### 1. Audio Description Accuracy (Weight: 2x)")
        st.markdown(
            """
        **What it measures:** How well the system describes what's actually happening in the audio
        - **1 - Poor:** Major inaccuracies, describes elements not present
        - **2 - Below Average:** Some correct elements but significant errors
        - **3 - Average:** Generally accurate but misses important details
        - **4 - Good:** Accurate with minor omissions or slight inaccuracies
        - **5 - Excellent:** Highly accurate, captures key audio elements precisely
        """
        )
        audio_description = st.selectbox(
            "Audio Description Accuracy", [1, 2, 3, 4, 5], key="audio_desc"
        )

        st.markdown("#### 2. Practical Advice Quality (Weight: 2x)")
        st.markdown(
            """
        **What it measures:** How actionable and useful the feedback is for improvement
        - **1 - Poor:** Vague, unhelpful, or incorrect advice
        - **2 - Below Average:** Some useful points but mostly generic
        - **3 - Average:** Decent advice but lacks specificity
        - **4 - Good:** Clear, actionable suggestions with some specifics
        - **5 - Excellent:** Highly specific, actionable advice tailored to the audio
        """
        )
        practical_advice = st.selectbox(
            "Practical Advice Quality", [1, 2, 3, 4, 5], key="practical"
        )

        st.markdown("#### 3. Relevance to Input Audio (Weight: 1.5x)")
        st.markdown(
            """
        **What it measures:** How well the feedback addresses the specific track analyzed
        - **1 - Poor:** Feedback could apply to any track
        - **2 - Below Average:** Mostly generic with minimal track-specific elements
        - **3 - Average:** Some track-specific feedback mixed with generic advice
        - **4 - Good:** Clearly addresses this specific track's characteristics
        - **5 - Excellent:** Highly tailored feedback that clearly references unique aspects
        """
        )
        relevance = st.selectbox(
            "Relevance to Input Audio", [1, 2, 3, 4, 5], key="relevance"
        )

        st.markdown("#### 4. Truthfulness (No Hallucinations) (Weight: 2x)")
        st.markdown(
            """
        **What it measures:** How truthful the system is, avoiding false or made-up information
        - **1 - Poor:** Frequent false claims, invented details
        - **2 - Below Average:** Some fabricated elements or unsupported claims
        - **3 - Average:** Occasional minor inaccuracies or assumptions
        - **4 - Good:** Mostly truthful with rare minor inaccuracies
        - **5 - Excellent:** No apparent fabrications, stays within bounds of evidence
        """
        )
        truthfulness = st.selectbox(
            "Truthfulness (No Hallucinations)", [1, 2, 3, 4, 5], key="truth"
        )

        st.markdown("#### 5. Technical Understanding (Weight: 1x)")
        st.markdown(
            """
        **What it measures:** Demonstrates understanding of arrangement patterns and audio features
        - **1 - Poor:** Shows no understanding of song structure or audio characteristics
        - **2 - Below Average:** Basic understanding but frequent errors
        - **3 - Average:** Decent grasp of concepts with some gaps
        - **4 - Good:** Good technical understanding with minor gaps
        - **5 - Excellent:** Strong technical understanding, uses arrangement info effectively
        """
        )
        technical = st.selectbox(
            "Technical Understanding", [1, 2, 3, 4, 5], key="technical"
        )

        # Notes section
        st.markdown("### Additional Notes")
        worked_well = st.text_area("What worked well:", height=80)
        key_issues = st.text_area("Key issues:", height=80)
        specific_examples = st.text_area("Specific examples of problems:", height=100)

        # Submit button
        submitted = st.form_submit_button("Save Evaluation", type="primary")

        if submitted:
            if not llm_output.strip():
                st.error("Please paste the LLM output to evaluate")
            elif not input_filename.strip():
                st.error("Please enter the input track filename")
            else:
                # Calculate weighted score
                weights = {
                    "audio_description": 2.0,
                    "practical_advice": 2.0,
                    "relevance": 1.5,
                    "truthfulness": 2.0,
                    "technical": 1.0,
                }

                weighted_score = (
                    audio_description * weights["audio_description"]
                    + practical_advice * weights["practical_advice"]
                    + relevance * weights["relevance"]
                    + truthfulness * weights["truthfulness"]
                    + technical * weights["technical"]
                )

                max_score = sum(weights.values()) * 5  # Max possible score
                percentage = (weighted_score / max_score) * 100

                # Create evaluation record
                eval_id = len(evaluations) + 1
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

                # Clean filename for use in file path
                clean_input_name = "".join(
                    c for c in input_filename.strip() if c.isalnum() or c in ("-", "_")
                )[:30]

                evaluation = {
                    "id": eval_id,
                    "timestamp": datetime.now().isoformat(),
                    "input_filename": input_filename.strip(),
                    "reference_filename": (
                        reference_filename.strip() if reference_filename else None
                    ),
                    "track_id": track_id.strip() if track_id else None,
                    "date": date.isoformat(),
                    "model_version": model_version.strip(),
                    "llm_output": llm_output.strip(),
                    "full_prompt": full_prompt.strip() if full_prompt else None,
                    "scores": {
                        "audio_description_accuracy": audio_description,
                        "practical_advice_quality": practical_advice,
                        "relevance_to_input": relevance,
                        "truthfulness": truthfulness,
                        "technical_understanding": technical,
                    },
                    "weighted_score": round(weighted_score, 2),
                    "max_score": max_score,
                    "percentage": round(percentage, 1),
                    "notes": {
                        "worked_well": worked_well.strip(),
                        "key_issues": key_issues.strip(),
                        "specific_examples": specific_examples.strip(),
                    },
                }

                # Create individual evaluation file
                eval_filename = (
                    f"eval_{eval_id:03d}_{timestamp_str}_{clean_input_name}.json"
                )
                eval_file_path = eval_dir / eval_filename

                # Save individual evaluation file
                try:
                    with open(eval_file_path, "w") as f:
                        json.dump(evaluation, f, indent=2)

                    st.success(
                        f"✅ Evaluation saved to `{eval_filename}`! Score: {weighted_score}/{max_score} ({percentage}%)"
                    )

                    # Show score breakdown
                    st.markdown("### Score Breakdown:")
                    st.markdown(
                        f"- Audio Description Accuracy: {audio_description}/5 (×{weights['audio_description']}) = {audio_description * weights['audio_description']}"
                    )
                    st.markdown(
                        f"- Practical Advice Quality: {practical_advice}/5 (×{weights['practical_advice']}) = {practical_advice * weights['practical_advice']}"
                    )
                    st.markdown(
                        f"- Relevance to Input Audio: {relevance}/5 (×{weights['relevance']}) = {relevance * weights['relevance']}"
                    )
                    st.markdown(
                        f"- Truthfulness: {truthfulness}/5 (×{weights['truthfulness']}) = {truthfulness * weights['truthfulness']}"
                    )
                    st.markdown(
                        f"- Technical Understanding: {technical}/5 (×{weights['technical']}) = {technical * weights['technical']}"
                    )
                    st.markdown(
                        f"**Total: {weighted_score}/{max_score} ({percentage}%)**"
                    )

                    st.rerun()
                except Exception as e:
                    st.error(f"Error saving evaluation: {e}")

    # Display existing evaluations
    if evaluations:
        st.markdown("---")
        st.markdown("### Evaluation History")

        # Summary stats
        avg_score = sum(e["percentage"] for e in evaluations) / len(evaluations)
        st.metric(
            "Average Score", f"{avg_score:.1f}%", f"{len(evaluations)} evaluations"
        )

        # Score over time plot
        if len(evaluations) >= 2:
            st.markdown("#### Score Trend Over Time")

            # Prepare data for plotting
            plot_data = []
            for i, eval_data in enumerate(evaluations, 1):
                plot_data.append(
                    {
                        "evaluation_number": i,
                        "percentage": eval_data["percentage"],
                        "date": eval_data["date"],
                        "input_filename": eval_data.get("input_filename", f"Eval {i}"),
                    }
                )

            df = pd.DataFrame(plot_data)

            # Create the plot with smaller font sizes
            plt.rcParams.update({"font.size": 10})
            fig, ax = plt.subplots(figsize=(10, 2))
            ax.plot(
                df["evaluation_number"],
                df["percentage"],
                "o-",
                color="#1f77b4",
                linewidth=2,
                markersize=4,
            )
            ax.set_xlabel("Evaluation Number", fontsize=10)
            ax.set_ylabel("Score (%)", fontsize=10)
            ax.set_title("RAG Output Evaluation Scores Over Time", fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 100)
            ax.tick_params(axis="both", labelsize=9)

            # Add trend line if we have enough data points
            if len(evaluations) >= 3:
                import numpy as np

                z = np.polyfit(df["evaluation_number"], df["percentage"], 1)
                p = np.poly1d(z)
                ax.plot(
                    df["evaluation_number"],
                    p(df["evaluation_number"]),
                    "--",
                    color="red",
                    alpha=0.7,
                    label=f"Trend (slope: {z[0]:.1f}%/eval)",
                )
                ax.legend()

            st.pyplot(fig)
            plt.close(fig)

        # Recent evaluations table
        st.markdown("#### Recent Evaluations")
        for eval_data in reversed(evaluations[-10:]):  # Show last 10
            # Create title with input filename or track_id
            title_id = eval_data.get(
                "input_filename", eval_data.get("track_id", "Unknown")
            )
            with st.expander(
                f"ID {eval_data['id']} - {title_id} - {eval_data['percentage']}% ({eval_data['date']})"
            ):

                col1, col2 = st.columns([1, 1])

                with col1:
                    st.markdown("**Scores:**")
                    for metric, score in eval_data["scores"].items():
                        st.markdown(f"- {metric.replace('_', ' ').title()}: {score}/5")
                    st.markdown(
                        f"**Weighted Total: {eval_data['weighted_score']}/{eval_data['max_score']} ({eval_data['percentage']}%)**"
                    )

                with col2:
                    st.markdown("**Details:**")
                    st.markdown(
                        f"**Input File:** {eval_data.get('input_filename', 'N/A')}"
                    )
                    if eval_data.get("reference_filename"):
                        st.markdown(
                            f"**Reference File:** {eval_data['reference_filename']}"
                        )
                    if eval_data.get("track_id"):
                        st.markdown(f"**Track ID:** {eval_data['track_id']}")
                    st.markdown(f"**Model:** {eval_data['model_version']}")
                    st.markdown(f"**Date:** {eval_data['date']}")
                    if eval_data["notes"]["key_issues"]:
                        st.markdown(f"**Issues:** {eval_data['notes']['key_issues']}")

                col_btn1, col_btn2, col_btn3 = st.columns(3)
                with col_btn1:
                    if st.button(f"Show LLM Output", key=f"show_{eval_data['id']}"):
                        st.text_area(
                            "LLM Output:",
                            eval_data["llm_output"],
                            height=200,
                            key=f"output_{eval_data['id']}",
                        )

                with col_btn2:
                    if eval_data.get("full_prompt") and st.button(
                        f"Show Prompt", key=f"show_prompt_{eval_data['id']}"
                    ):
                        st.text_area(
                            "Full Prompt Used:",
                            eval_data["full_prompt"],
                            height=300,
                            key=f"prompt_{eval_data['id']}",
                        )

                with col_btn3:
                    # Find previous evaluation with a prompt for diff
                    prev_eval_with_prompt = None
                    current_idx = None

                    # Find current evaluation index
                    for i, e in enumerate(evaluations):
                        if e["id"] == eval_data["id"]:
                            current_idx = i
                            break

                    # Find previous evaluation with a prompt
                    if current_idx is not None and current_idx > 0:
                        for i in range(current_idx - 1, -1, -1):
                            if evaluations[i].get("full_prompt"):
                                prev_eval_with_prompt = evaluations[i]
                                break

                    # Show diff button only if both current and previous have prompts
                    if eval_data.get("full_prompt") and prev_eval_with_prompt:
                        if st.button(
                            f"Diff vs #{prev_eval_with_prompt['id']}",
                            key=f"diff_{eval_data['id']}",
                        ):
                            st.markdown(
                                f"**Prompt Diff: #{prev_eval_with_prompt['id']} → #{eval_data['id']}**"
                            )

                            # Create unified diff
                            prev_lines = prev_eval_with_prompt[
                                "full_prompt"
                            ].splitlines(keepends=True)
                            curr_lines = eval_data["full_prompt"].splitlines(
                                keepends=True
                            )

                            diff = list(
                                difflib.unified_diff(
                                    prev_lines,
                                    curr_lines,
                                    fromfile=f"Evaluation #{prev_eval_with_prompt['id']}",
                                    tofile=f"Evaluation #{eval_data['id']}",
                                    lineterm="",
                                )
                            )

                            if diff:
                                diff_text = "".join(diff)
                                st.code(diff_text, language="diff")
                            else:
                                st.info("No differences found between prompts.")
                    else:
                        st.write("")  # Empty space to maintain column alignment

        # Export option
        if st.button("📊 Export All Evaluations"):
            # Create combined export data
            export_data = {
                "export_timestamp": datetime.now().isoformat(),
                "total_evaluations": len(evaluations),
                "average_score": (
                    sum(e["percentage"] for e in evaluations) / len(evaluations)
                    if evaluations
                    else 0
                ),
                "evaluations": evaluations,
            }

            st.download_button(
                label="Download Combined JSON",
                data=json.dumps(export_data, indent=2),
                file_name=f"feedback_evaluations_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
            )
    else:
        st.info(
            "No evaluations recorded yet. Complete the form above to start tracking feedback quality!"
        )

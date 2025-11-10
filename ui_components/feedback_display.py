import streamlit as st
from typing import Optional


def render_feedback(
    feedback: Optional[str],
    feedback_error: Optional[str],
    visual_only: bool = False,
):
    """
    Render AI feedback section.

    Args:
        feedback: Generated feedback text (None if not generated yet)
        feedback_error: Error message if feedback generation failed
        visual_only: Whether visual-only mode is enabled
    """
    st.markdown("**AI Music Mentor Feedback**")

    if not visual_only:
        if feedback_error:
            st.error(f"Could not generate feedback: {feedback_error}")
            st.info(
                "This might be because there are no training examples in the database yet."
            )
        else:
            st.markdown(feedback)
    else:
        st.success("Visual-only mode enabled")
        st.info(
            "AI feedback skipped for faster processing. Uncheck 'Visual Only' in the sidebar to get feedback."
        )

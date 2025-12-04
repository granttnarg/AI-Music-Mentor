import streamlit as st
from pathlib import Path


def load_css(css_file_path: str = "static/styles.css"):
    """
    Load and inject CSS into Streamlit app.

    Args:
        css_file_path: Path to the CSS file relative to project root
    """
    css_path = Path(css_file_path)

    if not css_path.exists():
        st.warning(f"CSS file not found: {css_file_path}")
        return

    with open(css_path) as f:
        css_content = f.read()
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)

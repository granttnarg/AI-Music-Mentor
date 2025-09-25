import streamlit as st
from admin_tabs.add_new import show_add_new_tab
from admin_tabs.browse_edit import show_browse_edit_tab
from admin_tabs.admin_eval import show_admin_eval_tab
from admin_tabs.similarity_viz import show_similarity_viz_tab

st.set_page_config(page_title="RAG Database Admin", layout="wide")

st.title("RAG Database Admin")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📤 Add New Training Example",
        "📋 Browse & Edit Examples",
        "📊 Evaluation Tracker",
        "🎯 3D Similarity Viz",
    ]
)

with tab1:
    show_add_new_tab()

with tab2:
    show_browse_edit_tab()

with tab3:
    show_admin_eval_tab()

with tab4:
    show_similarity_viz_tab()

import streamlit as st
from admin_tabs.add_new import show_add_new_tab
from admin_tabs.browse_edit import show_browse_edit_tab

st.set_page_config(page_title="RAG Database Admin", layout="wide")

st.title("RAG Database Admin")

# Create tabs
tab1, tab2 = st.tabs(["📤 Add New Training Example", "📋 Browse & Edit Examples"])

with tab1:
    show_add_new_tab()

with tab2:
    show_browse_edit_tab()

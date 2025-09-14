import streamlit as st
from pathlib import Path
import os
from dotenv import load_dotenv
from db.db import AudioRAGDatabase
from db.operations import AudioRAGOperations

load_dotenv()


@st.cache_resource
def get_database():
    """Initialize and return database connection"""
    connection_url = os.getenv(
        "DB_CONNECTION_URL", "postgresql://postgres:<ADD_TOENV_FILE>"
    )
    db = AudioRAGDatabase(connection_url)
    return AudioRAGOperations(db)


def show_browse_edit_tab():
    """Show the Browse & Edit Training Examples tab content"""
    st.markdown("#### Browse & Edit Training Examples")
    st.caption("View and edit existing training examples in the database")

    GENRES = [
        "techno",
        "deep techno",
        "hard techno",
        "broken techno",
        "tech-House",
        "house",
        "electro",
        "vocal techno",
        "ambient",
        "other",
    ]

    # Get all training examples
    try:
        db_ops = get_database()
        training_examples = db_ops.get_all_training_examples()

        if not training_examples:
            st.info(
                "No training examples found. Add some using the 'Add New' tab or batch import script."
            )
        else:
            st.success(f"Found {len(training_examples)} training examples")

            # Search and filter options
            col1, col2, col3 = st.columns(3)
            with col1:
                genre_filter = st.selectbox(
                    "Filter by genre:", ["All"] + GENRES, key="genre_filter"
                )
            with col2:
                search_query = st.text_input(
                    "Search in filenames:",
                    placeholder="Enter filename to search...",
                    key="search_query",
                )
            with col3:
                # Show only entries with placeholder feedback
                show_placeholders_only = st.checkbox(
                    "Show only placeholder feedback",
                    help="Show entries that need manual editing",
                )

            # Filter examples
            filtered_examples = training_examples

            # Genre filter
            if genre_filter != "All":
                filtered_examples = [
                    ex for ex in filtered_examples if ex["genre"] == genre_filter
                ]

            # Search filter
            if search_query.strip():
                search_lower = search_query.lower().strip()
                filtered_examples = [
                    ex
                    for ex in filtered_examples
                    if (
                        search_lower in ex["input_track"]["file_path"].lower()
                        or search_lower in ex["reference_track"]["file_path"].lower()
                    )
                ]

            # Placeholder feedback filter
            if show_placeholders_only:
                filtered_examples = [
                    ex
                    for ex in filtered_examples
                    if any("[EDIT ME]" in fb["text"] for fb in ex["feedback_items"])
                ]

            st.markdown(
                f"**Showing {len(filtered_examples)} of {len(training_examples)} examples**"
            )

            if len(filtered_examples) == 0:
                st.info(
                    "No examples match your filters. Try adjusting the search criteria."
                )

            # Display examples
            for i, example in enumerate(filtered_examples):
                with st.expander(
                    f"ID {example['id']} - {example['genre']} - {example['created_at'].strftime('%Y-%m-%d %H:%M')}"
                ):
                    # Basic info
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("**Input Track:**")
                        st.text(
                            f"File: {Path(example['input_track']['file_path']).name}"
                        )
                        st.text(f"Duration: {example['input_track']['duration']:.1f}s")

                        # Audio player for input track
                        if Path(example["input_track"]["file_path"]).exists():
                            st.audio(example["input_track"]["file_path"])
                        else:
                            st.warning("Input file not found")

                    with col2:
                        st.markdown("**Reference Track:**")
                        st.text(
                            f"File: {Path(example['reference_track']['file_path']).name}"
                        )
                        st.text(
                            f"Duration: {example['reference_track']['duration']:.1f}s"
                        )

                        # Audio player for reference track
                        if Path(example["reference_track"]["file_path"]).exists():
                            st.audio(example["reference_track"]["file_path"])
                        else:
                            st.warning("Reference file not found")

                    # Edit functionality
                    st.markdown("**Edit Training Example:**")

                    # Genre editing
                    current_genre = example["genre"]
                    new_genre = st.selectbox(
                        "Genre:",
                        GENRES,
                        index=(
                            GENRES.index(current_genre)
                            if current_genre in GENRES
                            else 0
                        ),
                        key=f"genre_{example['id']}",
                    )

                    # Quick edit for placeholder feedback
                    has_placeholder = any(
                        "[EDIT ME]" in fb["text"] for fb in example["feedback_items"]
                    )
                    if has_placeholder:
                        st.warning(
                            "⚠️ This entry has placeholder feedback that needs editing!"
                        )

                    # Feedback editing
                    st.markdown("**Feedback Items:**")

                    # Display existing feedback for editing
                    feedback_updates = []
                    feedback_types = [
                        "general",
                        "rhythm",
                        "rhythm_practical",
                        "eq",
                        "eq_practical",
                    ]

                    for j, feedback in enumerate(example["feedback_items"]):
                        with st.container():
                            col1, col2 = st.columns([3, 1])

                            with col1:
                                st.markdown(f"**Feedback {j+1}:**")

                            with col2:
                                # Delete button for this feedback item
                                delete_fb = st.button(
                                    "🗑️ Delete",
                                    key=f"delete_{example['id']}_{j}",
                                    help="Delete this feedback item",
                                )

                            if not delete_fb:  # Only include if not marked for deletion
                                type_index = 0
                                if feedback["type"] in feedback_types:
                                    type_index = feedback_types.index(feedback["type"])

                                fb_type = st.selectbox(
                                    "Type:",
                                    feedback_types,
                                    index=type_index,
                                    key=f"fb_type_{example['id']}_{j}",
                                )

                                fb_text = st.text_area(
                                    "Feedback text:",
                                    value=feedback["text"],
                                    height=100,
                                    key=f"fb_text_{example['id']}_{j}",
                                )

                                feedback_updates.append(
                                    {
                                        "id": feedback["id"],
                                        "type": fb_type,
                                        "text": fb_text,
                                    }
                                )
                            else:
                                st.success(
                                    "✅ This feedback will be deleted when you save changes"
                                )

                            st.markdown("---")

                    # Add new feedback option
                    st.markdown("**Add New Feedback:**")
                    add_new = st.checkbox(
                        f"Add new feedback item", key=f"add_new_{example['id']}"
                    )

                    if add_new:
                        new_fb_type = st.selectbox(
                            "New feedback type:",
                            [
                                "general",
                                "rhythm",
                                "rhythm_practical",
                                "eq",
                                "eq_practical",
                            ],
                            key=f"new_fb_type_{example['id']}",
                        )
                        new_fb_text = st.text_area(
                            "New feedback text:",
                            placeholder="Enter your feedback...",
                            height=100,
                            key=f"new_fb_text_{example['id']}",
                        )

                        if new_fb_text.strip():
                            feedback_updates.append(
                                {"type": new_fb_type, "text": new_fb_text}
                            )

                    # Save changes button
                    if st.button(
                        f"Save Changes", key=f"save_{example['id']}", type="primary"
                    ):
                        try:
                            genre_to_update = (
                                new_genre if new_genre != current_genre else None
                            )
                            db_ops.update_training_example_feedback(
                                example["id"], feedback_updates, genre_to_update
                            )
                            st.success("✅ Changes saved successfully!")
                            st.rerun()  # Refresh the page
                        except Exception as e:
                            st.error(f"❌ Error saving changes: {e}")

                    st.markdown("---")

    except Exception as e:
        st.error(f"Error loading training examples: {e}")

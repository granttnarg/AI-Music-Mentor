import streamlit as st

st.title("AI Music Mentor Dashboard")
st.markdown(
    "#### Upload your unfinished track to get helpful advice on how to finish it."
)
st.caption(
    "Our AI system is built from data of one experienced producers' feedback and gives advice in their subjective tone and style."
)

# File upload for MP3
uploaded_file = st.file_uploader("Upload Unfinished track - MP3 file", type=["mp3"])

# Text input
text_input = st.text_input("What do you need help with on your track?:")

# Dropdown with 3 options
dropdown_option = st.selectbox(
    "Stage your track is at:", ["Sketch", "Half Finished", "Almost Finished"]
)


if st.button("Submit"):

    if uploaded_file is not None:
        st.success(f"File uploaded: {uploaded_file.name}")
    else:
        st.warning("No file uploaded")

    if text_input:
        st.write(f"Text entered: {text_input}")
    else:
        st.warning("No text entered")

    st.write(f"Selected option: {dropdown_option}")

    # TOD: PROCESS UPLOAD HERE
    # Make sure we handle the files and send them to our pipeline
    # For processing.

    st.success("Form submitted successfully!")

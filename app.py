import streamlit as st
import joblib
# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Movie Genre Predictor",
    page_icon="🎬",
    layout="centered"
)

# -----------------------------
# Load trained model
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("movie_genre_model.joblib")

model = load_model()

# -----------------------------
# App title
# -----------------------------
st.title("🎬 Movie Genre Predictor")
st.write("Enter a movie title and description to predict its genre.")

st.divider()

# -----------------------------
# User input
# -----------------------------
title = st.text_input(
    "Movie Title",
    placeholder="Enter movie title..."
)

description = st.text_area(
    "Movie Description",
    placeholder="Enter the movie plot/description...",
    height=180
)

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔮 Predict Genre", use_container_width=True):

    if not title.strip() and not description.strip():
        st.warning("Please enter a movie title or description.")

    else:
        text = title + " " + description

        prediction = model.predict([text])[0]

        st.success(f"Predicted Genre: **{prediction.upper()}**")

# -----------------------------
# Footer
# -----------------------------
st.divider()

st.caption("Movie Genre Classification using TF-IDF + Linear SVM")

import os
import streamlit as st
import joblib
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from openai import OpenAI

load_dotenv()

app = Flask(__name__)

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise RuntimeError("OPENAI_API_KEY is not set.")

client = OpenAI(api_key=api_key)


@app.route("/")
def home():
    return render_template("index.html")

@st.cache_resource
def load_model():
    return joblib.load("movie_genre_model.joblib")

model = load_model()


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}

    movie_title = data.get("title", "").strip()
    description = data.get("description", "").strip()

    if not movie_title and not description:
        return jsonify({
            "error": "Please enter a movie title or description."
        }), 400

    prompt = f"""
You are a movie genre classification system.

Predict the most likely movie genre from the information below.

Movie title:
{movie_title or "Not provided"}

Movie description:
{description or "Not provided"}

Return ONLY valid JSON in this exact format:
{{
  "genre": "Genre name",
  "confidence": 0,
  "reason": "Short explanation"
}}

The confidence must be an integer from 0 to 100.
"""

    try:
        response = client.responses.create(
            model="gpt-5-mini",
            input=prompt
        )

        result_text = response.output_text.strip()

        import json
        result = json.loads(result_text)

        return jsonify(result)

    except json.JSONDecodeError:
        return jsonify({
            "error": "The AI returned an unexpected response."
        }), 500

    except Exception as e:
        print("AI error:", e)

        return jsonify({
            "error": "Unable to predict the genre right now."
        }), 500
st.divider()

st.caption("Movie Genre Classification using TF-IDF + Linear SVM")

if __name__ == "__main__":
    app.run(debug=True)
import streamlit as st
import joblib
import pandas as pd
import numpy as np


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Movie Genre Predictor",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM STYLING
# ============================================================

st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 50%, #f8fafc 100%);
        }

        .block-container {
            max-width: 1250px;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        .hero-box {
            padding: 2.5rem 2rem;
            border-radius: 24px;
            background: linear-gradient(135deg, #111827, #312e81, #4f46e5);
            color: white;
            margin-bottom: 2rem;
            box-shadow: 0 15px 40px rgba(49, 46, 129, 0.22);
        }

        .hero-small {
            font-size: 0.85rem;
            font-weight: 700;
            letter-spacing: 2px;
            opacity: 0.85;
            margin-bottom: 0.7rem;
        }

        .hero-title {
            font-size: 3rem;
            font-weight: 800;
            line-height: 1.1;
            margin-bottom: 1rem;
        }

        .hero-description {
            font-size: 1.05rem;
            line-height: 1.7;
            max-width: 800px;
            opacity: 0.92;
        }

        .metric-card {
            padding: 1.2rem;
            border-radius: 18px;
            background: white;
            border: 1px solid #e5e7eb;
            box-shadow: 0 8px 25px rgba(15, 23, 42, 0.06);
            text-align: center;
        }

        .metric-icon {
            font-size: 1.7rem;
            margin-bottom: 0.4rem;
        }

        .metric-title {
            font-size: 0.8rem;
            color: #64748b;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .metric-value {
            font-size: 1.25rem;
            font-weight: 800;
            color: #111827;
            margin-top: 0.2rem;
        }

        .section-title {
            font-size: 1.7rem;
            font-weight: 800;
            color: #111827;
            margin-top: 2rem;
            margin-bottom: 0.3rem;
        }

        .section-subtitle {
            color: #64748b;
            margin-bottom: 1.2rem;
        }

        .model-card {
            padding: 1.3rem;
            border-radius: 18px;
            background: white;
            border: 1px solid #e5e7eb;
            box-shadow: 0 7px 22px rgba(15, 23, 42, 0.05);
            min-height: 160px;
        }

        .model-name {
            font-size: 1.05rem;
            font-weight: 800;
            color: #111827;
            margin-bottom: 0.6rem;
        }

        .prediction-label {
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #64748b;
            font-weight: 700;
        }

        .prediction-value {
            font-size: 1.8rem;
            font-weight: 900;
            color: #4f46e5;
            margin-top: 0.2rem;
        }

        .explanation-card {
            padding: 1.2rem;
            border-radius: 16px;
            background: white;
            border: 1px solid #e5e7eb;
            margin-bottom: 0.8rem;
        }

        .positive-title {
            color: #15803d;
            font-weight: 800;
            font-size: 1rem;
        }

        .negative-title {
            color: #b91c1c;
            font-weight: 800;
            font-size: 1rem;
        }

        .feature-item {
            padding: 0.55rem 0.75rem;
            margin: 0.35rem 0;
            border-radius: 10px;
            background: #f8fafc;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .feature-name {
            font-weight: 650;
            color: #334155;
        }

        .feature-score {
            font-family: monospace;
            font-weight: 700;
        }

        .info-box {
            padding: 1rem 1.2rem;
            border-radius: 14px;
            background: #eef2ff;
            border-left: 4px solid #4f46e5;
            color: #3730a3;
            margin: 0.8rem 0 1rem 0;
        }

        .warning-box {
            padding: 1rem 1.2rem;
            border-radius: 14px;
            background: #fff7ed;
            border-left: 4px solid #f97316;
            color: #9a3412;
            margin: 0.8rem 0 1rem 0;
        }

        .footer {
            text-align: center;
            padding: 2rem 0 1rem 0;
            color: #64748b;
            font-size: 0.85rem;
        }

        div[data-testid="stButton"] > button {
            border-radius: 12px;
            font-weight: 750;
            min-height: 3rem;
        }

        div[data-testid="stTextInput"] input,
        div[data-testid="stTextArea"] textarea {
            border-radius: 12px;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():
    naive_bayes_model = joblib.load("movie_genre_naive_bayes.joblib")
    logistic_model = joblib.load("movie_genre_logistic_regression.joblib")
    svm_model = joblib.load("movie_genre_svm.joblib")

    return (
        naive_bayes_model,
        logistic_model,
        svm_model
    )


try:
    (
        naive_bayes_model,
        logistic_model,
        svm_model
    ) = load_models()

except Exception as e:
    st.error("Unable to load the trained models.")
    st.exception(e)
    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.title("🎬 MovieAI")

    st.caption("Genre Classification System")

    st.markdown("---")

    st.markdown(
        """
        ### 🤖 About the System

        This application predicts the genre of a movie from its
        title and plot description using Natural Language Processing
        and Machine Learning.
        """
    )

    st.markdown("### 🔄 ML Pipeline")

    st.markdown(
        """
        **1.** 📝 Text Data  
        ↓  
        **2.** 🔤 TF-IDF  
        ↓  
        **3.** 🧠 Classifier  
        ↓  
        **4.** 🎬 Movie Genre
        """
    )

    st.markdown("---")

    st.markdown("### 🧠 Algorithms")

    st.markdown(
        """
        - 📊 Naive Bayes
        - 📈 Logistic Regression
        - ⚡ Support Vector Machine
        """
    )

    st.markdown("---")

    st.info(
        "The Explainable AI section shows which words or phrases "
        "most influenced the prediction."
    )


# ============================================================
# HERO SECTION
# ============================================================

st.markdown("### ✨ AI POWERED • NLP • MACHINE LEARNING")

st.title("🎬 Movie Genre Predictor")

st.write(
    "Discover the hidden genre behind any movie plot. "
    "Our machine learning system analyzes your movie "
    "description using TF-IDF and three powerful "
    "classification algorithms."
)


# ============================================================
# FEATURE CARDS
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-icon">🔤</div>
            <div class="metric-title">Text Representation</div>
            <div class="metric-value">TF-IDF</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-icon">🧠</div>
            <div class="metric-title">ML Models</div>
            <div class="metric-value">3 Algorithms</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-icon">🎯</div>
            <div class="metric-title">Output</div>
            <div class="metric-value">Movie Genre</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# MOVIE INPUT
# ============================================================

st.markdown(
    '<div class="section-title">🎥 Movie Information</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-subtitle">Enter a movie title and its plot description to predict the genre.</div>',
    unsafe_allow_html=True
)

movie_title = st.text_input(
    "Movie Title",
    placeholder="Example: The Dark Knight"
)

model_option = st.selectbox(
    "Choose Prediction Mode",
    [
        "Compare All Models",
        "Naive Bayes",
        "Logistic Regression",
        "Support Vector Machine"
    ]
)

movie_plot = st.text_area(
    "Movie Plot / Description",
    height=180,
    placeholder=(
        "Example: A masked vigilante fights a dangerous criminal "
        "who spreads chaos throughout Gotham City..."
    )
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def predict(model, text):
    """Return model prediction."""
    return model.predict([text])[0]


def get_probabilities(model, text):
    """Return class probabilities when supported."""
    if not hasattr(model, "predict_proba"):
        return None

    try:
        probabilities = model.predict_proba([text])[0]
        classes = model.classes_

        result = pd.DataFrame(
            {
                "Genre": classes,
                "Probability": probabilities
            }
        )

        return result.sort_values(
            "Probability",
            ascending=False
        ).reset_index(drop=True)

    except Exception:
        return None


def get_svm_scores(model, text):
    """Return SVM decision scores."""
    try:
        scores = model.decision_function([text])
        classes = model.classes_

        scores = np.asarray(scores)

        if scores.ndim == 2:
            scores = scores[0]

        result = pd.DataFrame(
            {
                "Genre": classes,
                "Decision Score": scores
            }
        )

        return result.sort_values(
            "Decision Score",
            ascending=False
        ).reset_index(drop=True)

    except Exception:
        return None


# ============================================================
# PIPELINE EXTRACTION
# ============================================================

def get_pipeline_parts(model):
    """
    Extract the TF-IDF vectorizer and classifier from the
    saved sklearn Pipeline.
    """

    try:
        if not hasattr(model, "named_steps"):
            return None, None

        vectorizer = model.named_steps.get("tfidf")
        classifier = model.named_steps.get("classifier")

        return vectorizer, classifier

    except Exception:
        return None, None


# ============================================================
# FEATURE WEIGHTS
# ============================================================

def get_feature_weights(classifier, prediction):
    """
    Extract feature weights associated with the predicted class.

    Supported classifiers:
    - MultinomialNB
    - LogisticRegression
    - LinearSVC
    """

    try:
        classes = list(classifier.classes_)
        predicted_index = classes.index(prediction)

        # ---------------------------------------------
        # Naive Bayes
        # ---------------------------------------------

        if hasattr(classifier, "feature_log_prob_"):

            log_prob = classifier.feature_log_prob_

            predicted_log_prob = log_prob[predicted_index]

            other_indices = [
                i
                for i in range(len(classes))
                if i != predicted_index
            ]

            if len(other_indices) > 0:
                other_log_prob = log_prob[
                    other_indices
                ].mean(axis=0)

                weights = (
                    predicted_log_prob
                    - other_log_prob
                )

            else:
                weights = predicted_log_prob

            return weights

        # ---------------------------------------------
        # Logistic Regression / LinearSVC
        # ---------------------------------------------

        if hasattr(classifier, "coef_"):

            coefficients = classifier.coef_

            # Binary classification
            if len(classes) == 2:

                if predicted_index == 1:
                    return coefficients[0]

                return -coefficients[0]

            # Multiclass classification
            return coefficients[predicted_index]

    except Exception:
        return None

    return None


# ============================================================
# EXPLAINABLE AI
# ============================================================

def explain_prediction(
    model,
    text,
    prediction,
    top_n=10
):
    """
    Explain a prediction using the TF-IDF vectorizer stored
    inside the trained Pipeline.

    If the input contains vocabulary words known to the model,
    return input-specific feature contributions.

    If there are no matching vocabulary features, return the
    strongest general learned features for the predicted class.
    """

    vectorizer, classifier = get_pipeline_parts(model)

    if vectorizer is None or classifier is None:
        return {
            "success": False,
            "message": (
                "The saved model does not expose the expected "
                "TF-IDF vectorizer and classifier."
            )
        }

    try:
        # Transform using the SAME vectorizer used during training
        X = vectorizer.transform([text])

        feature_names = vectorizer.get_feature_names_out()

        weights = get_feature_weights(
            classifier,
            prediction
        )

        if weights is None:
            return {
                "success": False,
                "message": (
                    "Feature weights are not available for "
                    "this model."
                )
            }

        # ------------------------------------------------
        # Use only non-zero TF-IDF features
        # ------------------------------------------------

        nonzero_indices = X.indices
        nonzero_values = X.data

        recognized_count = len(nonzero_indices)

        # ------------------------------------------------
        # No vocabulary matches
        # ------------------------------------------------

        if recognized_count == 0:

            general_df = pd.DataFrame(
                {
                    "Feature": feature_names,
                    "Contribution": weights
                }
            )

            positive = (
                general_df[
                    general_df["Contribution"] > 0
                ]
                .sort_values(
                    "Contribution",
                    ascending=False
                )
                .head(top_n)
            )

            negative = (
                general_df[
                    general_df["Contribution"] < 0
                ]
                .sort_values(
                    "Contribution",
                    ascending=True
                )
                .head(top_n)
            )

            return {
                "success": True,
                "type": "general",
                "recognized_count": 0,
                "positive": positive,
                "negative": negative
            }

        # ------------------------------------------------
        # Input-specific explanations
        # ------------------------------------------------

        contributions = (
            nonzero_values
            * weights[nonzero_indices]
        )

        feature_df = pd.DataFrame(
            {
                "Feature": feature_names[nonzero_indices],
                "TF-IDF": nonzero_values,
                "Contribution": contributions
            }
        )

        positive = (
            feature_df[
                feature_df["Contribution"] > 0
            ]
            .sort_values(
                "Contribution",
                ascending=False
            )
            .head(top_n)
        )

        negative = (
            feature_df[
                feature_df["Contribution"] < 0
            ]
            .sort_values(
                "Contribution",
                ascending=True
            )
            .head(top_n)
        )

        return {
            "success": True,
            "type": "input",
            "recognized_count": recognized_count,
            "positive": positive,
            "negative": negative
        }

    except Exception as e:

        return {
            "success": False,
            "message": str(e)
        }


# ============================================================
# EXPLANATION DISPLAY
# ============================================================

def display_feature_list(
    dataframe,
    positive=True
):
    """Display feature contributions."""

    if dataframe is None or dataframe.empty:
        st.caption("No features available to display.")
        return

    for _, row in dataframe.iterrows():

        feature = str(row["Feature"])
        contribution = float(row["Contribution"])

        if positive:
            score_text = f"+{contribution:.4f}"
        else:
            score_text = f"{contribution:.4f}"

        st.markdown(
            f"""
            <div class="feature-item">
                <span class="feature-name">🔹 {feature}</span>
                <span class="feature-score">{score_text}</span>
            </div>
            """,
            unsafe_allow_html=True
        )


def display_explanation(
    model,
    text,
    prediction,
    model_name
):
    """Display Explainable AI results."""

    explanation = explain_prediction(
        model,
        text,
        prediction
    )

    if not explanation["success"]:
        st.warning(
            f"Feature explanation is not available for "
            f"{model_name}."
        )

        if "message" in explanation:
            st.caption(explanation["message"])

        return

    # ------------------------------------------------
    # General explanation
    # ------------------------------------------------

    if explanation["type"] == "general":

        st.markdown(
            """
            <div class="warning-box">
                <strong>⚠️ No matching TF-IDF vocabulary found</strong><br><br>
                The entered text did not contain words or phrases
                present in the model's learned TF-IDF vocabulary.
                Therefore, the features below are the strongest
                features learned by the model for this genre,
                rather than words found in the current input.
            </div>
            """,
            unsafe_allow_html=True
        )

    # ------------------------------------------------
    # Input-specific explanation
    # ------------------------------------------------

    else:

        count = explanation["recognized_count"]

        st.markdown(
            f"""
            <div class="info-box">
                <strong>🔎 Input-specific explanation</strong><br>
                The model recognized <strong>{count}</strong>
                TF-IDF feature(s) from your movie description.
                These features contributed directly to the prediction.
            </div>
            """,
            unsafe_allow_html=True
        )

    positive_col, negative_col = st.columns(2)

    # ------------------------------------------------
    # Positive features
    # ------------------------------------------------

    with positive_col:

        st.markdown(
            """
            <div class="explanation-card">
                <div class="positive-title">
                    🟢 Features Supporting the Prediction
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        display_feature_list(
            explanation["positive"],
            positive=True
        )

    # ------------------------------------------------
    # Negative features
    # ------------------------------------------------

    with negative_col:

        st.markdown(
            """
            <div class="explanation-card">
                <div class="negative-title">
                    🔴 Features Working Against the Prediction
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        display_feature_list(
            explanation["negative"],
            positive=False
        )


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.markdown("")

predict_button = st.button(
    "🚀 Predict Movie Genre",
    use_container_width=True,
    type="primary"
)


# ============================================================
# VALIDATION
# ============================================================

if predict_button:

    if not movie_title.strip() and not movie_plot.strip():

        st.warning(
            "Please enter a movie title or movie plot."
        )
        st.stop()

    # Combine title and description
    combined_text = (
        movie_title.strip()
        + " "
        + movie_plot.strip()
    ).strip()

    st.markdown("---")


    # ========================================================
    # COMPARE ALL MODELS
    # ========================================================

    if model_option == "Compare All Models":

        nb_prediction = predict(
            naive_bayes_model,
            combined_text
        )

        lr_prediction = predict(
            logistic_model,
            combined_text
        )

        svm_prediction = predict(
            svm_model,
            combined_text
        )

        predictions = [
            nb_prediction,
            lr_prediction,
            svm_prediction
        ]

        vote_counts = (
            pd.Series(predictions)
            .value_counts()
        )

        final_genre = vote_counts.index[0]

        # ----------------------------------------------------
        # Final Prediction
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">🎯 Final Prediction</div>',
            unsafe_allow_html=True
        )

        st.success(
            f"Predicted Movie Genre: **{final_genre.upper()}**"
        )

        # ----------------------------------------------------
        # Model Results
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">🧠 Model Predictions</div>',
            unsafe_allow_html=True
        )

        col1, col2, col3 = st.columns(3)

        model_results = [
            (
                col1,
                "📊 Naive Bayes",
                nb_prediction
            ),
            (
                col2,
                "📈 Logistic Regression",
                lr_prediction
            ),
            (
                col3,
                "⚡ Support Vector Machine",
                svm_prediction
            )
        ]

        for column, name, prediction in model_results:

            with column:

                st.markdown(
                    f"""
                    <div class="model-card">
                        <div class="model-name">{name}</div>
                        <div class="prediction-label">
                            Predicted Genre
                        </div>
                        <div class="prediction-value">
                            {prediction}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # ----------------------------------------------------
        # Agreement
        # ----------------------------------------------------

        if len(set(predictions)) == 1:

            st.success(
                "✅ All three models agree on the prediction."
            )

        else:

            st.info(
                "ℹ️ The models produced different predictions. "
                "The final genre is selected using majority voting."
            )

        # ====================================================
        # PREDICTION ANALYSIS
        # ====================================================

        st.markdown(
            '<div class="section-title">📊 Prediction Analysis</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="section-subtitle">Explore probability estimates and decision scores from each model.</div>',
            unsafe_allow_html=True
        )

        tab1, tab2, tab3 = st.tabs(
            [
                "📊 Naive Bayes",
                "📈 Logistic Regression",
                "⚡ SVM"
            ]
        )

        # ----------------------------------------------------
        # Naive Bayes
        # ----------------------------------------------------

        with tab1:

            probabilities = get_probabilities(
                naive_bayes_model,
                combined_text
            )

            if probabilities is not None:

                st.dataframe(
                    probabilities,
                    use_container_width=True,
                    hide_index=True
                )

                chart_data = probabilities.set_index(
                    "Genre"
                )

                st.bar_chart(
                    chart_data["Probability"]
                )

            else:

                st.info(
                    "Probability scores are not available."
                )

        # ----------------------------------------------------
        # Logistic Regression
        # ----------------------------------------------------

        with tab2:

            probabilities = get_probabilities(
                logistic_model,
                combined_text
            )

            if probabilities is not None:

                st.dataframe(
                    probabilities,
                    use_container_width=True,
                    hide_index=True
                )

                chart_data = probabilities.set_index(
                    "Genre"
                )

                st.bar_chart(
                    chart_data["Probability"]
                )

            else:

                st.info(
                    "Probability scores are not available."
                )

        # ----------------------------------------------------
        # SVM
        # ----------------------------------------------------

        with tab3:

            scores = get_svm_scores(
                svm_model,
                combined_text
            )

            if scores is not None:

                st.dataframe(
                    scores,
                    use_container_width=True,
                    hide_index=True
                )

                chart_data = scores.set_index(
                    "Genre"
                )

                st.bar_chart(
                    chart_data["Decision Score"]
                )

            else:

                st.info(
                    "Decision scores are not available."
                )

        # ====================================================
        # EXPLAINABLE AI
        # ====================================================

        st.markdown(
            '<div class="section-title">🔍 Explainable AI</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="section-subtitle">Understand which TF-IDF words and phrases influenced each model.</div>',
            unsafe_allow_html=True
        )

        explain_tab1, explain_tab2, explain_tab3 = st.tabs(
            [
                "📊 Naive Bayes Explanation",
                "📈 Logistic Regression Explanation",
                "⚡ SVM Explanation"
            ]
        )

        with explain_tab1:

            display_explanation(
                naive_bayes_model,
                combined_text,
                nb_prediction,
                "Naive Bayes"
            )

        with explain_tab2:

            display_explanation(
                logistic_model,
                combined_text,
                lr_prediction,
                "Logistic Regression"
            )

        with explain_tab3:

            display_explanation(
                svm_model,
                combined_text,
                svm_prediction,
                "Support Vector Machine"
            )


    # ========================================================
    # SINGLE MODEL
    # ========================================================

    else:

        if model_option == "Naive Bayes":

            selected_model = naive_bayes_model

        elif model_option == "Logistic Regression":

            selected_model = logistic_model

        else:

            selected_model = svm_model


        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        prediction = predict(
            selected_model,
            combined_text
        )

        st.markdown(
            '<div class="section-title">🎯 Prediction Result</div>',
            unsafe_allow_html=True
        )

        st.success(
            f"Predicted Movie Genre: **{prediction.upper()}**"
        )


        # ----------------------------------------------------
        # Prediction Analysis
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">📊 Prediction Analysis</div>',
            unsafe_allow_html=True
        )

        if model_option in [
            "Naive Bayes",
            "Logistic Regression"
        ]:

            probabilities = get_probabilities(
                selected_model,
                combined_text
            )

            if probabilities is not None:

                st.dataframe(
                    probabilities,
                    use_container_width=True,
                    hide_index=True
                )

                chart_data = probabilities.set_index(
                    "Genre"
                )

                st.bar_chart(
                    chart_data["Probability"]
                )

            else:

                st.info(
                    "Probability scores are not available."
                )

        else:

            scores = get_svm_scores(
                selected_model,
                combined_text
            )

            if scores is not None:

                st.dataframe(
                    scores,
                    use_container_width=True,
                    hide_index=True
                )

                chart_data = scores.set_index(
                    "Genre"
                )

                st.bar_chart(
                    chart_data["Decision Score"]
                )

            else:

                st.info(
                    "Decision scores are not available."
                )


        # ----------------------------------------------------
        # Explainable AI
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">🔍 Explainable AI</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="section-subtitle">See which learned TF-IDF features support or oppose this prediction.</div>',
            unsafe_allow_html=True
        )

        display_explanation(
            selected_model,
            combined_text,
            prediction,
            model_option
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        🎬 Movie Genre Predictor &nbsp;•&nbsp;
        Powered by NLP, TF-IDF & Machine Learning
        <br>
        Naive Bayes • Logistic Regression • Support Vector Machine
    </div>
    """,
    unsafe_allow_html=True
)
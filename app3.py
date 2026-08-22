# ============================================

import streamlit as st
import joblib
import pandas as pd


# ============================================
# 1. PAGE CONFIG
# ============================================

st.set_page_config(
    page_title="Movie Genre Predictor",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================
# 2. LOAD MODELS
# ============================================

@st.cache_resource
def load_models():

    nb_model = joblib.load(
        "movie_genre_naive_bayes.joblib"
    )

    lr_model = joblib.load(
        "movie_genre_logistic_regression.joblib"
    )

    svm_model = joblib.load(
        "movie_genre_svm.joblib"
    )

    return nb_model, lr_model, svm_model


try:

    (
        nb_model,
        lr_model,
        svm_model
    ) = load_models()

except FileNotFoundError:

    st.error(
        "Model files not found. "
        "Please run your training script first."
    )

    st.stop()


# ============================================
# 3. CUSTOM CSS
# ============================================

st.markdown(
    """
    <style>

    /* ================================
       GLOBAL
       ================================ */

    .stApp {
        background:
            radial-gradient(
                circle at 10% 10%,
                rgba(120, 80, 200, 0.18),
                transparent 30%
            ),
            radial-gradient(
                circle at 90% 20%,
                rgba(255, 80, 130, 0.12),
                transparent 30%
            ),
            #080b14;
        color: #f5f5f7;
    }


    /* ================================
       HIDE STREAMLIT DEFAULT
       ================================ */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        background: transparent !important;
    }


    /* ================================
       SIDEBAR
       ================================ */

    section[data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #0d1020 0%,
                #090b13 100%
            );

        border-right: 1px solid
            rgba(255,255,255,0.08);
    }

    section[data-testid="stSidebar"] * {
        color: #e8e8ef;
    }


    /* ================================
       MAIN CONTAINER
       ================================ */

    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }


    # /* ================================
    #    HERO
    #    ================================ */

    # .hero {
    #     padding: 45px 40px;
    #     border-radius: 28px;

    #     background:
    #         linear-gradient(
    #             135deg,
    #             rgba(116, 72, 255, 0.32),
    #             rgba(236, 72, 153, 0.18)
    #         );

    #     border: 1px solid
    #         rgba(255,255,255,0.10);

    #     box-shadow:
    #         0 25px 80px
    #         rgba(0,0,0,0.35);

    #     margin-bottom: 30px;
    # }

    # .hero-badge {
    #     display: inline-block;

    #     padding: 7px 14px;

    #     border-radius: 50px;

    #     background: rgba(255,255,255,0.10);

    #     border: 1px solid
    #         rgba(255,255,255,0.12);

    #     color: #d9ccff;

    #     font-size: 13px;

    #     font-weight: 600;

    #     letter-spacing: 0.5px;

    #     margin-bottom: 15px;
    # }

    # .hero-title {
    #     font-size: 48px;
    #     font-weight: 800;

    #     line-height: 1.1;

    #     margin: 0;

    #     background:
    #         linear-gradient(
    #             90deg,
    #             #ffffff,
    #             #c9b8ff,
    #             #ffb4d4
    #         );

    #     -webkit-background-clip: text;
    #     -webkit-text-fill-color: transparent;
    # }

    # .hero-text {
    #     color: #b9b9c8;

    #     font-size: 17px;

    #     margin-top: 15px;

    #     max-width: 700px;

    #     line-height: 1.6;
    # }

    

/*========================================
   CINEMATIC HERO
 ======================================== */

    .hero {

        position: relative;

        min-height: 380px;

        padding: 55px 55px;

        border-radius: 30px;

        overflow: hidden;

        background:
            radial-gradient(
                circle at 85% 25%,
                rgba(236,72,153,0.35),
                transparent 28%
            ),
            radial-gradient(
                circle at 65% 80%,
                rgba(124,58,237,0.40),
                transparent 35%
            ),
            linear-gradient(
                135deg,
                #171329 0%,
                #21163c 45%,
                #120d20 100%
            );

        border: 1px solid
            rgba(255,255,255,0.12);

        box-shadow:
            0 30px 90px
            rgba(0,0,0,0.45);

        margin-bottom: 35px;

        display: flex;

        align-items: center;

        justify-content: space-between;
    }


    /* ========================================
       HERO CONTENT
       ======================================== */

    .hero-content {

        position: relative;

        z-index: 5;

        max-width: 680px;
    }


    /* ========================================
       BADGE
       ======================================== */

    .hero-badge {

        display: inline-block;

        padding: 8px 16px;

        border-radius: 50px;

        background:
            rgba(255,255,255,0.08);

        border: 1px solid
            rgba(255,255,255,0.14);

        color: #d7c7ff;

        font-size: 11px;

        font-weight: 800;

        letter-spacing: 1.3px;

        margin-bottom: 18px;

        box-shadow:
            0 5px 20px
            rgba(0,0,0,0.15);
    }


    /* ========================================
       HERO TITLE
       ======================================== */

    .hero-title {

        font-size: 53px;

        line-height: 1.05;

        font-weight: 900;

        letter-spacing: -2px;

        color: #ffffff;

        margin-bottom: 20px;
    }


    .hero-title span {

        background:
            linear-gradient(
                90deg,
                #a78bfa,
                #e879f9,
                #fb7185
            );

        -webkit-background-clip: text;

        -webkit-text-fill-color: transparent;

        background-clip: text;
    }


    /* ========================================
       HERO DESCRIPTION
       ======================================== */

    .hero-text {

        color: #bcb8ca;

        font-size: 16px;

        line-height: 1.7;

        max-width: 650px;

        margin-bottom: 28px;
    }


    .hero-text b {

        color: #d8caff;
    }


    /* ========================================
       HERO FEATURES
       ======================================== */

    .hero-features {

        display: flex;

        gap: 12px;

        flex-wrap: wrap;
    }


    .hero-feature {

        display: flex;

        align-items: center;

        gap: 10px;

        padding: 10px 14px;

        border-radius: 12px;

        background:
            rgba(255,255,255,0.06);

        border: 1px solid
            rgba(255,255,255,0.08);

        min-width: 115px;
    }


    .hero-feature > span {

        font-size: 21px;
    }


    .hero-feature b {

        display: block;

        color: #ffffff;

        font-size: 12px;
    }


    .hero-feature small {

        display: block;

        color: #8f8ca0;

        font-size: 10px;

        margin-top: 2px;
    }


    /* ========================================
       HERO DECORATION
       ======================================== */

    .hero-decoration {

        position: relative;

        width: 280px;

        height: 280px;

        margin-right: 25px;
    }


    /* ========================================
       MAIN MOVIE CIRCLE
       ======================================== */

    .movie-circle {

        position: absolute;

        width: 190px;

        height: 190px;

        border-radius: 50%;

        top: 40px;

        left: 45px;

        display: flex;

        align-items: center;

        justify-content: center;

        font-size: 75px;

        background:
            radial-gradient(
                circle at 35% 30%,
                #9b7cff,
                #5b21b6 55%,
                #241044
            );

        border: 1px solid
            rgba(255,255,255,0.18);

        box-shadow:
            0 0 70px
            rgba(139,92,246,0.45),
            inset 0 0 40px
            rgba(255,255,255,0.08);
    }


    /* ========================================
       FLOATING GENRE CARDS
       ======================================== */

    .floating-card {

        position: absolute;

        padding: 10px 16px;

        border-radius: 12px;

        background:
            rgba(20,17,35,0.85);

        backdrop-filter: blur(12px);

        border: 1px solid
            rgba(255,255,255,0.12);

        color: #ffffff;

        font-size: 12px;

        font-weight: 700;

        box-shadow:
            0 12px 30px
            rgba(0,0,0,0.25);
    }


    .card-one {

        top: 20px;

        right: 0;

        transform: rotate(5deg);
    }


    .card-two {

        bottom: 28px;

        right: -10px;

        transform: rotate(-5deg);
    }


    .card-three {

        bottom: 5px;

        left: 5px;

        transform: rotate(4deg);
    }


    /* ========================================
       RESPONSIVE
       ======================================== */

    @media (max-width: 900px) {

        .hero {

            padding: 40px 30px;

        }

        .hero-title {

            font-size: 42px;

        }

        .hero-decoration {

            display: none;

        }

    }


    @media (max-width: 600px) {

        .hero {

            padding: 30px 22px;

            min-height: auto;

        }

        .hero-title {

            font-size: 36px;

            letter-spacing: -1px;

        }

        .hero-text {

            font-size: 14px;

        }

        .hero-feature {

            min-width: 105px;

        }

    }

 
    /* ================================
       SECTION TITLES
       ================================ */

    .section-title {
        font-size: 25px;
        font-weight: 750;

        margin-top: 25px;
        margin-bottom: 15px;

        color: #ffffff;
    }


    /* ================================
       INPUT LABELS
       ================================ */

    label {
        color: #d8d8e5 !important;
        font-weight: 600 !important;
    }


    /* ================================
       INPUT BOXES
       ================================ */

    div[data-baseweb="input"] > div,
    div[data-baseweb="textarea"] > div {

        background: #ffffff
            !important;

        border: 1px solid
            #d1d5db
            !important;

        border-radius: 14px
            !important;

        # color: Black
        #     !important;
    }

    div[data-baseweb="input"] input,
    div[data-baseweb="textarea"] textarea {
      color: #000000 !important;
      background: #ffffff !important;
      caret-color: #000000 !important;
    }

    div[data-baseweb="input"] input::placeholder,
    div[data-baseweb="textarea"] textarea::placeholder {
      color: #777777 !important;
      opacity: 1 !important;
    }

    div[data-baseweb="input"]:focus-within,
    div[data-baseweb="textarea"]:focus-within {

        border-color:
            rgba(157, 124, 255, 0.8)
            !important;

        box-shadow:
            0 0 0 2px
            rgba(124, 92, 255, 0.12)
            !important;
    }


    /* ================================
       SELECT BOX
       ================================ */

    div[data-baseweb="select"] > div {

        background:
            rgba(255,255,255,0.055)
            !important;

        border: 1px solid
            rgba(255,255,255,0.10)
            !important;

        border-radius: 14px
            !important;

        color: white
            !important;
    }


    /* ================================
       BUTTON
       ================================ */

    .stButton > button {

        width: 100%;

        min-height: 52px;

        border: none;

        border-radius: 14px;

        background:
            linear-gradient(
                90deg,
                #7048ff,
                #a855f7,
                #ec4899
            );

        color: white;

        font-size: 16px;

        font-weight: 700;

        box-shadow:
            0 10px 30px
            rgba(124, 58, 237, 0.30);

        transition: all 0.25s ease;
    }

    .stButton > button:hover {

        transform: translateY(-2px);

        box-shadow:
            0 15px 40px
            rgba(236, 72, 153, 0.30);
    }


    /* ================================
       RESULT CARD
       ================================ */

    .result-card {

        padding: 30px;

        border-radius: 22px;

        background:
            linear-gradient(
                145deg,
                rgba(255,255,255,0.075),
                rgba(255,255,255,0.025)
            );

        border: 1px solid
            rgba(255,255,255,0.10);

        box-shadow:
            0 20px 60px
            rgba(0,0,0,0.25);

        text-align: center;

        margin-top: 20px;
    }

    .result-label {

        color: #aaaabd;

        font-size: 14px;

        text-transform: uppercase;

        letter-spacing: 2px;

        font-weight: 700;
    }

    .result-genre {

        font-size: 42px;

        font-weight: 800;

        margin-top: 8px;

        background:
            linear-gradient(
                90deg,
                #b9a2ff,
                #ff9ac5
            );

        -webkit-background-clip: text;

        -webkit-text-fill-color: transparent;
    }


    /* ================================
       MODEL CARDS
       ================================ */

    .model-card {

        padding: 24px;

        min-height: 155px;

        border-radius: 20px;

        background:
            rgba(255,255,255,0.045);

        border: 1px solid
            rgba(255,255,255,0.08);

        text-align: center;

        transition: 0.2s ease;
    }

    .model-name {

        font-size: 15px;

        color: #a8a8ba;

        font-weight: 600;

        margin-bottom: 10px;
    }

    .model-result {

        font-size: 27px;

        font-weight: 800;

        color: #ffffff;
    }


    /* ================================
       INFO CARDS
       ================================ */

    .info-card {

        padding: 20px;

        border-radius: 18px;

        background:
            rgba(255,255,255,0.045);

        border: 1px solid
            rgba(255,255,255,0.08);

        margin-top: 10px;

        color: #c6c6d4;

        line-height: 1.6;
    }


    /* ================================
       SIDEBAR BRAND
       ================================ */

    .side-brand {

        text-align: center;

        padding: 20px 0 30px;
    }

    .side-icon {

        font-size: 48px;
    }

    .side-title {

        font-size: 22px;

        font-weight: 800;

        margin-top: 5px;
    }

    .side-subtitle {

        color: #858597;

        font-size: 13px;
    }


    /* ================================
       FOOTER
       ================================ */

    .footer {

        text-align: center;

        color: #666679;

        font-size: 13px;

        margin-top: 50px;

        padding-top: 20px;

        border-top: 1px solid
            rgba(255,255,255,0.07);
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================
# 4. SIDEBAR
# ============================================

with st.sidebar:

    st.markdown(
        """
        <div class="side-brand">

            <div class="side-icon">🎬</div>

            <div class="side-title">
                MovieAI
            </div>

            <div class="side-subtitle">
                Genre Classification System
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown("### 🧠 Machine Learning")

    st.write(
        "This application predicts the genre "
        "of a movie from its title and plot."
    )

    st.markdown("### ⚙️ Pipeline")

    st.write("📚 Text Data")
    st.write("↓")
    st.write("🔤 TF-IDF")
    st.write("↓")
    st.write("🤖 Classifier")
    st.write("↓")
    st.write("🎬 Movie Genre")

    st.markdown("---")

    st.markdown("### 🚀 Algorithms")

    st.write("• Naive Bayes")
    st.write("• Logistic Regression")
    st.write("• Support Vector Machine")

    st.markdown("---")

    st.caption(
        "Built with Python • Scikit-learn • Streamlit"
    )

# ============================================
# 🎬 CINEMATIC HERO SECTION
# ============================================

st.markdown(
    """
    <div class="hero">

        <div class="hero-content">

            <div class="hero-badge">
                ✨ AI POWERED • NLP • MACHINE LEARNING
            </div>

            <div class="hero-title">
                🎬 Movie Genre
                <br>
                <span>Predictor</span>
            </div>

            <div class="hero-text">
                Discover the hidden genre behind any movie plot.
                Our machine learning system analyzes your movie
                description using <b>TF-IDF</b> and three powerful
                classification algorithms.
            </div>

            <div class="hero-features">

                <div class="hero-feature">
                    <span>🔤</span>
                    <div>
                        <b>TF-IDF</b>
                        <small>Text Features</small>
                    </div>
                </div>

                <div class="hero-feature">
                    <span>🤖</span>
                    <div>
                        <b>3 Models</b>
                        <small>ML Classifiers</small>
                    </div>
                </div>

                <div class="hero-feature">
                    <span>🎯</span>
                    <div>
                        <b>Genre</b>
                        <small>Prediction</small>
                    </div>
                </div>

            </div>

        </div>

        <div class="hero-decoration">

            <div class="movie-circle">
                🎞️
            </div>

            <div class="floating-card card-one">
                🎭 Drama
            </div>

            <div class="floating-card card-two">
                ⚡ Action
            </div>

            <div class="floating-card card-three">
                😂 Comedy
            </div>

        </div>

    </div>
    """,
    unsafe_allow_html=True
)
# # ============================================
# # 5. HERO
# # ============================================

# st.markdown(
#     """
#     <div class="hero">

#         <div class="hero-badge">
#             ✨ MACHINE LEARNING PROJECT
#         </div>

#         <h1 class="hero-title">
#             Movie Genre Predictor
#         </h1>

#         <div class="hero-text">
#             Discover the most likely genre of any movie
#             using Natural Language Processing and
#             traditional Machine Learning.
#             Enter a title and plot summary to get started.
#         </div>

#     </div>
#     """,
#     unsafe_allow_html=True
# )


# ============================================
# 6. MOVIE INPUT
# ============================================

st.markdown(
    '<div class="section-title">🎞️ Movie Information</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns([1, 2], gap="large")

with col1:

    movie_title = st.text_input(
        "Movie Title",
        placeholder="The Dark Knight"
    )


with col2:

    model_choice = st.selectbox(
        "Classification Model",
        [
            "Compare All Models",
            "Naive Bayes",
            "Logistic Regression",
            "Support Vector Machine"
        ]
    )


movie_plot = st.text_area(
    "Plot Summary / Description",
    placeholder=(
        "Example: A masked vigilante fights a dangerous "
        "criminal who spreads chaos throughout Gotham City..."
    ),
    height=180
)


st.write("")


# ============================================
# 7. PREDICT BUTTON
# ============================================

predict_clicked = st.button(
    "✨ Predict Movie Genre",
    use_container_width=True
)


# ============================================
# 8. HELPER FUNCTIONS
# ============================================

def predict(model, text):

    return model.predict([text])[0]


def get_probabilities(model, text):

    if not hasattr(model, "predict_proba"):
        return None

    probabilities = model.predict_proba([text])[0]

    classes = model.classes_

    result = pd.DataFrame({
        "Genre": classes,
        "Probability": probabilities
    })

    return result.sort_values(
        "Probability",
        ascending=False
    )


# ============================================
# 9. PREDICTION
# ============================================
if predict_clicked:

    if not movie_title.strip() and not movie_plot.strip():
        st.warning(
            "🎞️ Please enter a movie title or plot summary."
        )
        st.stop()

    text = (
        movie_title.strip()
        + " "
        + movie_plot.strip()
    )

    if model_choice == "Compare All Models":

        # Naive Bayes
        nb_prediction = nb_model.predict([text])[0]

        # Logistic Regression
        lr_prediction = lr_model.predict([text])[0]

        # SVM
        svm_prediction = svm_model.predict([text])[0]

        # Majority vote
        predictions = [
            nb_prediction,
            lr_prediction,
            svm_prediction
        ]

        counts = pd.Series(predictions).value_counts()

        final_genre = counts.index[0]

        st.success(
            f"🏆 Final Predicted Genre: "
            f"{final_genre.title()}"
        )

    elif model_choice == "Naive Bayes":

        prediction = nb_model.predict([text])[0]

        st.success(
            f"🧮 Naive Bayes Prediction: "
            f"{prediction.title()}"
        )

    elif model_choice == "Logistic Regression":

        prediction = lr_model.predict([text])[0]

        st.success(
            f"📈 Logistic Regression Prediction: "
            f"{prediction.title()}"
        )

    elif model_choice == "Support Vector Machine":

        prediction = svm_model.predict([text])[0]

        st.success(
            f"⚡ SVM Prediction: "
            f"{prediction.title()}"
        )
# if predict_clicked:

#     if not movie_title.strip() and not movie_plot.strip():

#         st.warning(
#             "🎞️ Please enter a movie title or plot summary."
#         )

#         st.stop()


#     text = (
#         movie_title.strip()
#         + " "
#         + movie_plot.strip()
#     )


#     st.markdown(
#         '<div class="section-title">'
#         '🎯 Prediction Results'
#         '</div>',
#         unsafe_allow_html=True
#     )


    # ========================================
    # COMPARE ALL
    # ========================================

    if model_choice == "Compare All Models":

        nb_prediction = predict(
            nb_model,
            text
        )

        lr_prediction = predict(
            lr_model,
            text
        )

        svm_prediction = predict(
            svm_model,
            text
        )


        # ------------------------------------
        # Model cards
        # ------------------------------------

        c1, c2, c3 = st.columns(3, gap="medium")


        with c1:

            st.markdown(
                f"""
                <div class="model-card">

                    <div class="model-name">
                        🧮 NAIVE BAYES
                    </div>

                    <div class="model-result">
                        {nb_prediction.title()}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


        with c2:

            st.markdown(
                f"""
                <div class="model-card">

                    <div class="model-name">
                        📈 LOGISTIC REGRESSION
                    </div>

                    <div class="model-result">
                        {lr_prediction.title()}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


        with c3:

            st.markdown(
                f"""
                <div class="model-card">

                    <div class="model-name">
                        ⚡ SUPPORT VECTOR MACHINE
                    </div>

                    <div class="model-result">
                        {svm_prediction.title()}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


        # ------------------------------------
        # Voting
        # ------------------------------------

        predictions = [
            nb_prediction,
            lr_prediction,
            svm_prediction
        ]

        counts = pd.Series(
            predictions
        ).value_counts()

        final_genre = counts.index[0]

        votes = counts.iloc[0]


        st.markdown(
            f"""
            <div class="result-card">

                <div class="result-label">
                    🏆 FINAL PREDICTION
                </div>

                <div class="result-genre">
                    {final_genre.title()}
                </div>

                <p style="color:#aaaabd;">
                    {votes} of 3 machine-learning models
                    agree on this genre.
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )


        # ------------------------------------
        # Agreement
        # ------------------------------------

        if votes == 3:

            st.success(
                "🎯 Perfect agreement — all three models "
                "predicted the same genre."
            )

        elif votes == 2:

            st.info(
                "🤝 Majority agreement — two models "
                "predicted the same genre."
            )

        else:

            st.warning(
                "⚠️ The models disagree. "
                "Consider the individual predictions."
            )


        # ====================================
        # PROBABILITIES
        # ====================================

        st.markdown(
            '<div class="section-title">'
            '📊 Prediction Analysis'
            '</div>',
            unsafe_allow_html=True
        )


        tab1, tab2 = st.tabs([
            "🧮 Naive Bayes",
            "📈 Logistic Regression"
        ])


        with tab1:

            nb_probs = get_probabilities(
                nb_model,
                text
            )

            if nb_probs is not None:

                display = nb_probs.head(5).copy()

                display["Probability"] = (
                    display["Probability"] * 100
                )

                for _, row in display.iterrows():

                    genre = row["Genre"].title()

                    probability = row["Probability"]

                    st.write(
                        f"**{genre}** — "
                        f"{probability:.2f}%"
                    )

                    st.progress(
                        min(
                            float(probability / 100),
                            1.0
                        )
                    )


        with tab2:

            lr_probs = get_probabilities(
                lr_model,
                text
            )

            if lr_probs is not None:

                display = lr_probs.head(5).copy()

                display["Probability"] = (
                    display["Probability"] * 100
                )

                for _, row in display.iterrows():

                    genre = row["Genre"].title()

                    probability = row["Probability"]

                    st.write(
                        f"**{genre}** — "
                        f"{probability:.2f}%"
                    )

                    st.progress(
                        min(
                            float(probability / 100),
                            1.0
                        )
                    )


        # ====================================
        # SVM
        # ====================================

        st.markdown(
            "### ⚡ SVM Decision Scores"
        )

        svm_scores = svm_model.decision_function(
            [text]
        )[0]

        svm_classes = svm_model.classes_

        svm_df = pd.DataFrame({
            "Genre": svm_classes,
            "Decision Score": svm_scores
        })

        svm_df = svm_df.sort_values(
            "Decision Score",
            ascending=False
        ).head(5)

        st.dataframe(
            svm_df,
            use_container_width=True,
            hide_index=True
        )


    # ========================================
    # SINGLE MODEL
    # ========================================

    else:

        if model_choice == "Naive Bayes":

            selected_model = nb_model

        elif model_choice == "Logistic Regression":

            selected_model = lr_model

        else:

            selected_model = svm_model


        prediction = predict(
            selected_model,
            text
        )


        # ------------------------------------
        # Main result
        # ------------------------------------

        st.markdown(
            f"""
            <div class="result-card">

                <div class="result-label">
                    PREDICTED MOVIE GENRE
                </div>

                <div class="result-genre">
                    🎬 {prediction.title()}
                </div>

                <p style="color:#aaaabd;">
                    Model: {model_choice}
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )


        # ------------------------------------
        # Probability models
        # ------------------------------------

        probabilities = get_probabilities(
            selected_model,
            text
        )


        if probabilities is not None:

            st.markdown(
                '<div class="section-title">'
                '📊 Genre Probabilities'
                '</div>',
                unsafe_allow_html=True
            )


            top_results = probabilities.head(5)


            for _, row in top_results.iterrows():

                genre = row["Genre"].title()

                probability = (
                    row["Probability"] * 100
                )

                st.write(
                    f"**{genre}** — "
                    f"{probability:.2f}%"
                )

                st.progress(
                    float(row["Probability"])
                )


        # ------------------------------------
        # SVM scores
        # ------------------------------------

        else:

            st.markdown(
                '<div class="section-title">'
                '⚡ SVM Decision Scores'
                '</div>',
                unsafe_allow_html=True
            )


            scores = selected_model.decision_function(
                [text]
            )[0]

            classes = selected_model.classes_


            score_df = pd.DataFrame({

                "Genre": classes,

                "Decision Score": scores

            })


            score_df = score_df.sort_values(
                "Decision Score",
                ascending=False
            ).head(5)


            st.dataframe(
                score_df,
                use_container_width=True,
                hide_index=True
            )


# ============================================
# 10. FOOTER
# ============================================

st.markdown(
    """
    <div class="footer">

        🎬 Movie Genre Predictor
        &nbsp;•&nbsp;
        TF-IDF
        &nbsp;•&nbsp;
        Naive Bayes
        &nbsp;•&nbsp;
        Logistic Regression
        &nbsp;•&nbsp;
        SVM

    </div>
    """,
    unsafe_allow_html=True
)
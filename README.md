"# CODSOFT_TASK1" 
# 🎬 Movie Genre Classification

> **An AI-powered NLP application that predicts a movie's genre from its title and plot description.**

## 📌 1. Project Overview

Movie Genre Classification is a Natural Language Processing (NLP) and Machine Learning project developed as part of **CODSOFT Task 1**.

The goal of this project is to automatically identify the most likely genre of a movie based on the information provided by the user.

The application combines the **movie title** and **plot description**, transforms the text into meaningful numerical features using **TF-IDF**, and then passes those features through trained machine learning models.

Three different classification algorithms are used:

- 🤖 Multinomial Naive Bayes
- 📈 Logistic Regression
- ⚡ Linear Support Vector Machine (SVM)

The project also includes an interactive **Streamlit web application** and an **Explainable AI (XAI)** feature that helps users understand which words or phrases influenced a prediction.

---

## ✨ 2. Features

The application provides a complete and interactive movie genre prediction experience.

### 🎥 Movie Information Input
Users can enter:

- Movie title
- Movie plot or description

The application combines both inputs before sending them to the trained NLP pipeline.

### 🤖 Multiple Machine Learning Models

Predictions can be generated using:

- **Naive Bayes**
- **Logistic Regression**
- **Support Vector Machine**

### ⚖️ Model Comparison

The **Compare All Models** mode runs the same movie through all three classifiers and displays their individual predictions.

A majority-vote approach is then used to determine the final predicted genre.

### 📊 Prediction Analysis

The application provides a visual analysis of the model predictions, making it easier to understand whether the models agree or disagree.

### 🧠 Explainable AI

The application goes beyond simply showing a prediction.

The Explainable AI section identifies important words and phrases that support or oppose the predicted genre, helping users understand the model's decision.

### 🎨 Interactive Streamlit Interface

A clean and user-friendly Streamlit interface makes the machine learning system easy to test without writing Python code.

---

## 🔄 3. ML/NLP Pipeline

The project follows a complete text-classification pipeline:

```text
                  Movie Title
                      +
                Plot Description
                      │
                      ▼
              Text Combination
                      │
                      ▼
               TF-IDF Vectorizer
                      │
                      ▼
             Numerical Text Features
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
     Naive Bayes  Logistic      Linear
                  Regression       SVM
          │           │           │
          └───────────┼───────────┘
                      ▼
                Genre Prediction
                      │
                      ▼
              Prediction Analysis
                      │
                      ▼
              Explainable AI

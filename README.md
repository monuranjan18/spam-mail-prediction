# 📧 Spam Mail Prediction

A machine learning project for detecting whether a text message is **Spam** or **Ham** using Natural Language Processing (NLP), TF-IDF feature extraction, and a Linear Support Vector Machine (SVM).

The project includes dataset preprocessing, exploratory data analysis, model comparison, hyperparameter tuning, model persistence, a command-line prediction interface, and an interactive Streamlit web application.

---

## 📌 Project Overview

Spam messages are unwanted or potentially harmful messages that may contain misleading offers, fraudulent links, advertisements, or other unwanted content.

This project builds a supervised machine learning classifier that learns from labeled SMS messages and predicts whether a new message is:

- **HAM** — legitimate/normal message
- **SPAM** — unwanted or potentially suspicious message

The complete workflow is:

```text
Raw Dataset
     ↓
Data Inspection
     ↓
Data Cleaning
     ↓
Exploratory Data Analysis
     ↓
Train/Test Split
     ↓
TF-IDF Feature Extraction
     ↓
Model Comparison
     ↓
Linear SVM Selection
     ↓
Model Evaluation
     ↓
Joblib Model Persistence
     ↓
CLI Prediction
     ↓
Streamlit Web Application
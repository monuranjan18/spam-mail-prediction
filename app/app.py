from pathlib import Path

import joblib
import streamlit as st


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "spam_classifier.joblib"


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Spam Mail Predictor",
    page_icon="📧",
    layout="centered"
)


# --------------------------------------------------
# Load model
# --------------------------------------------------

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


model = load_model()


# --------------------------------------------------
# UI
# --------------------------------------------------

st.title("📧 Spam Mail Predictor")

st.write(
    "Enter a message below and the trained machine learning "
    "model will classify it as **SPAM** or **HAM**."
)

st.info(
    "The model uses TF-IDF features with a Linear SVM classifier."
)

message = st.text_area(
    "Enter your message:",
    height=180,
    placeholder="Example: Congratulations! You have won a free prize..."
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("🔍 Predict", use_container_width=True):

    message = message.strip()

    if not message:
        st.warning("Please enter a message before predicting.")

    else:
        prediction = model.predict([message])[0]
        decision_score = model.decision_function([message])[0]

        if prediction == "spam":
            st.error("🚨 SPAM")
        else:
            st.success("✅ HAM")

        st.write(
            f"**Decision score:** `{decision_score:.4f}`"
        )

        st.caption(
            "The decision score indicates the model's position "
            "relative to its classification boundary. "
            "It is not a probability."
        )


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "Spam Mail Prediction • TF-IDF + Linear SVM"
)
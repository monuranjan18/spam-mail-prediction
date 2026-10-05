from pathlib import Path
import sys

import joblib


# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Saved model
MODEL_PATH = PROJECT_ROOT / "models" / "spam_classifier.joblib"


def load_model():
    """Load the trained spam classifier pipeline."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


def predict_message(message, model):
    """Predict whether a message is ham or spam."""
    prediction = model.predict([message])[0]

    # LinearSVC does not provide predict_proba().
    # decision_function gives a score, not a probability.
    decision_score = model.decision_function([message])[0]

    label = prediction.upper()

    return label, decision_score


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print('python src/predict.py "your message here"')
        return

    message = " ".join(sys.argv[1:]).strip()

    if not message:
        print("Error: Message cannot be empty.")
        return

    model = load_model()

    label, score = predict_message(message, model)

    print("\nSpam Mail Prediction")
    print("=" * 30)
    print(f"Message : {message}")
    print(f"Prediction: {label}")
    print(f"Decision score: {score:.4f}")


if __name__ == "__main__":
    main()
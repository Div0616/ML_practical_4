# Practical 4: Sentiment Analysis Classification using Naive Bayes

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.title("ML Practical 4 – Sentiment Analysis (Naive Bayes)")

# ─────────────────────────────────────────
# Step 1: Dataset (Positive=1 / Negative=0)
# ─────────────────────────────────────────
reviews = [
    # Positive
    "I absolutely loved this movie, it was fantastic",
    "Great performance by all the actors",
    "The storyline was brilliant and engaging",
    "One of the best films I have ever seen",
    "An outstanding masterpiece of cinema",
    "Really enjoyed every moment of the film",
    "Wonderful direction and superb acting",
    "Highly recommend this movie to everyone",
    "A heartwarming and uplifting experience",
    "Perfect film with an amazing ending",
    # Negative
    "This movie was a total waste of time",
    "Terrible acting and a boring plot",
    "I hated every second of this film",
    "Worst movie I have ever watched",
    "Very disappointing and poorly made",
    "The story made no sense at all",
    "Dull and completely uninteresting",
    "A disaster from start to finish",
    "Could not sit through the entire film",
    "Awful script and bad direction",
]

labels = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# ─────────────────────────────────────────
# Step 2: TF-IDF Vectorization
# ─────────────────────────────────────────
vectorizer = TfidfVectorizer(lowercase=True, stop_words='english', ngram_range=(1, 2))
X = vectorizer.fit_transform(reviews)

# ─────────────────────────────────────────
# Step 3: Train / Test Split
# ─────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, random_state=42, stratify=labels
)

# ─────────────────────────────────────────
# Step 4: Train Naive Bayes Model
# ─────────────────────────────────────────
model = MultinomialNB()
model.fit(X_train, y_train)

# ─────────────────────────────────────────
# Step 5: Evaluate
# ─────────────────────────────────────────
y_pred = model.predict(X_test)

st.subheader("Model Evaluation")
st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.4f}")
st.text("Classification Report:\n" + classification_report(
    y_test, y_pred, target_names=["Negative", "Positive"]
))

# ─────────────────────────────────────────
# Step 6: Confusion Matrix Plot
# ─────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"], ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix – Sentiment Analysis (Naive Bayes)")
plt.tight_layout()
st.pyplot(fig)

# ─────────────────────────────────────────
# Step 7: Predict on a New Review
# ─────────────────────────────────────────
def predict_sentiment(text):
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    return "Positive 😊" if pred == 1 else "Negative 😞"

st.subheader("Try the Classifier")
user_review = st.text_input("Enter a movie review:", placeholder="Type your review here...")

if user_review:
    sentiment = predict_sentiment(user_review)
    st.success(f"Predicted Sentiment: {sentiment}")

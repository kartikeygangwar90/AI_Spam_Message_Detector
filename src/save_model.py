import pandas as pd
import joblib 

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    f1_score,
    recall_score,
    confusion_matrix,
    classification_report
)

# Load preprocessed data
df = pd.read_csv("data/preprocessed_spam.csv")

# Input and target
x = df["Cleaned_message"].fillna("")
y = df["label"]

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# TF-IDF
vectorizer = TfidfVectorizer()

# Fit vectorizer only on training data
x_train_tfidf = vectorizer.fit_transform(x_train)


# Create logistic regression model
model = LinearSVC()

# Train model
model.fit(x_train_tfidf, y_train)


# save model
joblib.dump(model, "model/spam_model.pkl")

# save vectorizer
joblib.dump(vectorizer, "model/tfidf_vectorizer.pkl")

print("Model saved succesfully!")
print("Vectorizer Saved succesfully!")
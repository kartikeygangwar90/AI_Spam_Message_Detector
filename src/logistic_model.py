import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

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

x_train_tfidf = vectorizer.fit_transform(x_train)
x_test_tfidf = vectorizer.transform(x_test)

# Create logistic regression model
model = LogisticRegression(max_iter = 1000)

# Train model
model.fit(x_train_tfidf, y_train)

# Make predictions
y_pred = model.predict(x_test_tfidf)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Logistics Regression Evaluation")

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)

# confusion matrix 
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# Classification report
print("\nClassification report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Ham", "Spam"]
    )
)
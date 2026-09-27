import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load preprocessed data 
df = pd.read_csv("data/preprocessed_spam.csv")

# Input and target
x = df["Cleaned_message"].fillna("")
y = df["label"]


# train test split
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

# Create Naive Bayes model
model = MultinomialNB()

# train the model 
model.fit(x_train_tfidf, y_train)

print("Model training completed!")

print("Training samples :", x_train_tfidf.shape[0])
print("Features:", x_train_tfidf.shape[1])
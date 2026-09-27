import joblib

from processing import preprocess_text

# load saved model and vectorizer
model = joblib.load("model/spam_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

# test message
message = "Congratulations! You have won a free prize"

# preprocess text
Cleaned_message = preprocess_text(message)

# convert to TF-IDF
message_tfidf = vectorizer.transform([Cleaned_message])

# prediction
prediction = model.predict(message_tfidf)[0]

if prediction == 1:
    print("Prediction: SPAM")
else :
    print("Prediction: HAM")



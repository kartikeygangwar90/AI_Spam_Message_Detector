import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

#load preprocessed data  
df = pd.read_csv("data/preprocessed_spam.csv")

# Input targets and features
x = df["Cleaned_message"].fillna("")
y = df["label"]

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# create TF-IDF vectorization
vectorizer = TfidfVectorizer()

# learn vocabulary from training data
# and convert training messages into numbers
x_train_tfidf = vectorizer.fit_transform(x_train)

# convert test messages into same vectorizer
x_test_tfidf = vectorizer.transform(x_test)


# Display info 
print("Training TF_IDF shape:", x_train_tfidf.shape)
print("Testing TF_IDF shape:", x_test_tfidf.shape)

print("\nNumber of words in Vocabulary:")
print(len(vectorizer.vocabulary_))

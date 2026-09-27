import pandas as pd
import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

# Intializing NLP tools
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

# Main preprocessing function
def preprocess_text(text):
    # converting the text to lower case
    text = text.lower()

    # tokenize
    tokens = word_tokenize(text)

    # Removing punctuation, numbers and stopwords from tokens
    tokens = [
        word for word in tokens
        if word.isalpha() and word not in stop_words
    ]

    # Stem words : reducing  the size of words and similar words to more simple words 
    tokens = [
        stemmer.stem(word)
        for word in tokens
    ]

    # convert list back to text
    return " ".join(tokens)

# loading the cleaned dataset
# df = pd.read_csv("data/cleaned_spam.csv")

# Applying the preprocessing 
# df["Cleaned_message"] = df["Message"].apply(preprocess_text)

# # display example
# print(df[["Message", "Cleaned_message"]].head(10))

# # Save preprocessed dataset
# df.to_csv("data/preprocessed_spam.csv", index = False)

# print("\n Preprocessing completed")

# message = "Congratulations!!! You have WON a FREE prize!!!"

# print("Original Message : ")
# print(message)

# print("\nNltk processing : ")
# print(process_text(message))
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/preprocessed_spam.csv")

x = df["Cleaned_message"]
y = df["label"]

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Total Samples: ", len(df))
print("Training samples: ", len(x_train))
# print("Training samples: ", len(y_train))
print("Testing samples: ", len(x_test))
# print("Testing samples: ", len(y_test))

print("\nTraining level distrubution:")
print(y_train.value_counts(normalize=True))

print("\nTesting level distribution:")
print(y_test.value_counts(normalize=True))
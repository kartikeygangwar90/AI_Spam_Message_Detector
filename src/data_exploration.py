import pandas as pd

df = pd.read_csv(
    "data/SMSSpamCollection",
    sep = "\t",
    header=None,
    names=["label", "message"]
)

print("First 5 rows :")
print(df.head())

print("\nDataset shape")
print(df.shape)

print("\nLabel distribution")
print(df["label"].value_counts())

print("\nMissing values: ")
print(df.isnull().sum())

print("\nDuplicate rows: ")
print(df.duplicated().sum())

print("\n Smaple spam message: ")
print(df[df["label"] == "spam"]["message"].head(10))


print("\n Smaple Ham message: ")
print(df[df["label"] == "ham"]["message"].head(10))


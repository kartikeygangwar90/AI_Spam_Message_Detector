import pandas as pd

#load data sets
df = pd.read_csv(
    "data/SMSSpamCollection",
    sep = "\t",
    header = None,
    names = ["label", "Message"]
)

# Original Shape
print(df.head())

# Missing data
print("Missing Values:")
print(df.isnull().sum())

# Duplicate data
print("\nDuplicates: ")
print(df.duplicated().sum())

# Before its shape
print("Before removing duplicates: ", df.shape)

# To remove duplicates
df = df.drop_duplicates()

# To print the new shape after deleting duplicates
print("After removing duplicates", df.shape)

# Check labels
print("\nUnique Labels:")
print(df["label"].unique())

# Convert labels to Numerical values
df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})

# check datatypes
print(df.dtypes)

# Display cleaned data
print(df.head())

# Save cleaned data
df.to_csv("data/cleaned_data.csv", index = False)

print("\nCleaned dataset saved succesfully")
import pandas as pd
import seaborn as sns

# ---------------------------------------
# 1. Load Dataset
# ---------------------------------------

df = sns.load_dataset("titanic")

print("ORIGINAL DATASET SHAPE")
print(df.shape)

# ---------------------------------------
# 2. Remove unnecessary columns
# ---------------------------------------

# Deck has too many missing values, so we remove it
df = df.drop("deck", axis=1)

# ---------------------------------------
# 3. Handle missing values
# ---------------------------------------

# Fill missing Age values with median age
df["age"] = df["age"].fillna(df["age"].median())

# Fill missing Embarked values with the most common value
df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])

# Fill missing Embark Town values with the most common value
df["embark_town"] = df["embark_town"].fillna(
    df["embark_town"].mode()[0]
)

# ---------------------------------------
# 4. Remove duplicate rows
# ---------------------------------------

df = df.drop_duplicates()

# ---------------------------------------
# 5. Display cleaned dataset information
# ---------------------------------------

print("\nCLEANED DATASET SHAPE")
print(df.shape)

print("\nMISSING VALUES AFTER CLEANING")
print(df.isnull().sum())

print("\nDUPLICATES AFTER CLEANING")
print(df.duplicated().sum())

# ---------------------------------------
# 6. Statistical Summary
# ---------------------------------------

print("\nSTATISTICAL SUMMARY")
print(df.describe())

# ---------------------------------------
# 7. Survival Analysis
# ---------------------------------------

print("\nSURVIVAL COUNT")
print(df["survived"].value_counts())

print("\nSURVIVAL PERCENTAGE")
print(
    df["survived"].value_counts(normalize=True) * 100
)

# ---------------------------------------
# 8. Survival by Gender
# ---------------------------------------

print("\nSURVIVAL BY GENDER")
print(
    df.groupby("sex", observed=True)["survived"]
    .mean() * 100
)

# ---------------------------------------
# 9. Survival by Passenger Class
# ---------------------------------------

print("\nSURVIVAL BY PASSENGER CLASS")
print(
    df.groupby("class", observed=True)["survived"]
    .mean() * 100
)
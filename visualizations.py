import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------------------------------
# Create results folder
# ---------------------------------------

os.makedirs("results", exist_ok=True)

# ---------------------------------------
# Load Dataset
# ---------------------------------------

df = sns.load_dataset("titanic")

# ---------------------------------------
# Clean Dataset
# ---------------------------------------

df = df.drop("deck", axis=1)

df["age"] = df["age"].fillna(df["age"].median())

df["embarked"] = df["embarked"].fillna(
    df["embarked"].mode()[0]
)

df["embark_town"] = df["embark_town"].fillna(
    df["embark_town"].mode()[0]
)

df = df.drop_duplicates()

# ---------------------------------------
# 1. Survival Count
# ---------------------------------------

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="survived"
)

plt.title("Titanic Survival Count")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Number of Passengers")

plt.tight_layout()
plt.savefig("results/survival_count.png")
plt.show()

# ---------------------------------------
# 2. Survival by Gender
# ---------------------------------------

plt.figure(figsize=(7, 5))

sns.barplot(
    data=df,
    x="sex",
    y="survived"
)

plt.title("Survival Rate by Gender")
plt.xlabel("Gender")
plt.ylabel("Survival Rate")

plt.tight_layout()
plt.savefig("results/survival_by_gender.png")
plt.show()

# ---------------------------------------
# 3. Survival by Passenger Class
# ---------------------------------------

plt.figure(figsize=(7, 5))

sns.barplot(
    data=df,
    x="class",
    y="survived"
)

plt.title("Survival Rate by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")

plt.tight_layout()
plt.savefig("results/survival_by_class.png")
plt.show()

# ---------------------------------------
# 4. Age Distribution
# ---------------------------------------

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="age",
    bins=30,
    kde=True
)

plt.title("Age Distribution of Passengers")
plt.xlabel("Age")
plt.ylabel("Number of Passengers")

plt.tight_layout()
plt.savefig("results/age_distribution.png")
plt.show()

# ---------------------------------------
# 5. Fare Distribution
# ---------------------------------------

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="fare",
    bins=30,
    kde=True
)

plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Number of Passengers")

plt.tight_layout()
plt.savefig("results/fare_distribution.png")
plt.show()

# ---------------------------------------
# 6. Correlation Heatmap
# ---------------------------------------

numeric_columns = [
    "survived",
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]

correlation = df[numeric_columns].corr()

plt.figure(figsize=(8, 6))

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.tight_layout()
plt.savefig("results/correlation_heatmap.png")
plt.show()

# ---------------------------------------
# 7. Age vs Fare
# ---------------------------------------

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="age",
    y="fare",
    hue="survived"
)

plt.title("Age vs Fare by Survival")
plt.xlabel("Age")
plt.ylabel("Fare")

plt.tight_layout()
plt.savefig("results/age_vs_fare.png")
plt.show()

print("\nAll visualizations created successfully!")

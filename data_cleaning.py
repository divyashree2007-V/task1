
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set visualization style
sns.set_theme(style="whitegrid")

# Create folder for charts
os.makedirs("visualizations", exist_ok=True)

# 1. LOAD DATASET
df = pd.read_csv("dataset.csv")

print("========== ORIGINAL DATASET ==========")
print(df.head())

print("\nDataset Shape:", df.shape)
print("\nDataset Information:")
df.info()

# 2. CHECK MISSING VALUES
print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

# Handle missing values
for col in df.columns:
    if df[col].isnull().sum() > 0:
        if df[col].dtype == "object":
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())

print("\nMissing values handled successfully!")

# 3. REMOVE DUPLICATES
print("\nDuplicate Records:", df.duplicated().sum())

df = df.drop_duplicates()

print("Duplicates removed successfully!")

# 4. OUTLIER DETECTION AND HANDLING
# Absences is checked using the IQR method.

Q1 = df["absences"].quantile(0.25)
Q3 = df["absences"].quantile(0.75)

IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

outliers = df[
    (df["absences"] < lower_limit) |
    (df["absences"] > upper_limit)
]

print("\nOutliers in Absences:", len(outliers))

# Cap extreme absence values
df["absences"] = df["absences"].clip(
    lower=lower_limit,
    upper=upper_limit
)

print("Outlier handling completed!")

# 5. SAVE CLEANED DATASET
df.to_csv("cleaned_dataset.csv", index=False)

print("\nCleaned Dataset Saved Successfully!")

# 6. DATA VISUALIZATION

# Chart 1: Student Gender Distribution
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="sex")
plt.title("Student Gender Distribution")
plt.xlabel("Gender")
plt.ylabel("Number of Students")
plt.savefig("visualizations/gender_distribution.png")
plt.show()

# Chart 2: Final Grade Distribution
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="G3", bins=15, kde=True)
plt.title("Final Grade Distribution")
plt.xlabel("Final Grade")
plt.ylabel("Number of Students")
plt.savefig("visualizations/grade_distribution.png")
plt.show()

# Chart 3: Study Time vs Final Grade
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x="studytime", y="G3")
plt.title("Study Time vs Final Grade")
plt.xlabel("Study Time")
plt.ylabel("Average Final Grade")
plt.savefig("visualizations/studytime_vs_grade.png")
plt.show()

# Chart 4: Absences vs Final Grade
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="absences", y="G3")
plt.title("Absences vs Final Grade")
plt.xlabel("Number of Absences")
plt.ylabel("Final Grade")
plt.savefig("visualizations/absences_vs_grade.png")
plt.show()

# Chart 5: Average Grade by Gender
plt.figure(figsize=(7, 5))
sns.barplot(data=df, x="sex", y="G3")
plt.title("Average Final Grade by Gender")
plt.xlabel("Gender")
plt.ylabel("Average Final Grade")
plt.savefig("visualizations/average_grade_gender.png")
plt.show()

# Chart 6: Correlation Heatmap
plt.figure(figsize=(10, 7))
numeric_df = df.select_dtypes(include=np.number)

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Student Performance Correlation Heatmap")
plt.savefig("visualizations/correlation_heatmap.png")
plt.show()

# 7. KEY INSIGHTS
print("\n========== KEY INSIGHTS ==========")

print("Total Students:", len(df))

print("Average Final Grade:", round(df["G3"].mean(), 2))

print("Highest Final Grade:", df["G3"].max())

print("Lowest Final Grade:", df["G3"].min())

print("Average Study Time:", round(df["studytime"].mean(), 2))

print("Average Absences:", round(df["absences"].mean(), 2))

print("\nAverage Grade by Gender:")
print(df.groupby("sex")["G3"].mean())

print("\nAverage Grade by Study Time:")
print(df.groupby("studytime")["G3"].mean())

print("\n========== PROJECT COMPLETED ==========")
print("Data Cleaning and Visualization Completed Successfully!")

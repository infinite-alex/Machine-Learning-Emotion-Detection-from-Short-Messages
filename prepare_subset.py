import pandas as pd

df = pd.read_csv("data/raw/emotions.csv", sep=",")

print(df.columns)
print(df.head())

subset = df.groupby("label").sample(n=7000, random_state=42)

print(subset.columns)
print(subset["label"].value_counts())

subset.to_csv("data/processed/emotion_subset.csv", index=False)

print("saved")
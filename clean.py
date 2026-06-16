import re #regular expression cleaning the language
import nltk # natural language processing
import pandas as pd
from nltk.corpus import stopwords # sa taiem prepozitiile sau cuvintele de umplutura

#nltk.download("stopwords")
stop_words= set(stopwords.words("english"))
custom_stopwords = {
    "im", "ive", "id", "ill", "dont", "didnt",
    "doesnt", "cant", "couldnt", "wouldnt"
}
stop_words.update(custom_stopwords)
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    words = text.split()
    words= [word for word in words if word not in stop_words]
    return " ".join(words)

#example ="I FEEL unhappy when no one likes my fb post!!!"
#print(clean_text(example))

df = pd.read_csv("data/processed/emotion_subset.csv")
df["clean_text"] = df["text"].apply(clean_text)
print(df[["text", "clean_text"]].head())
df.to_csv("data/processed/emotion_clean.csv", index=False)

print("saved")
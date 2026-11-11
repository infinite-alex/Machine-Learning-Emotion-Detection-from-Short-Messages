"""Try the trained models on your own sentences.

Usage:
    python predict.py "I can't wait to see you tomorrow!"
    python predict.py            # interactive mode
"""
import re
import sys
import warnings

import joblib

warnings.filterwarnings("ignore")

EMOTIONS = {0: "sadness 😢", 1: "joy 😄", 2: "love ❤️", 3: "anger 😠", 4: "fear 😨", 5: "surprise 😲"}


# NLTK English stopwords (same list as clean.py), included so no download is needed
NLTK_STOPWORDS = """i me my myself we our ours ourselves you you're you've you'll you'd your yours yourself yourselves he him his himself she she's her hers herself it it's its itself they them their theirs themselves what which who whom this that that'll these those am is are was were be been being have has had having do does did doing a an the and but if or because as until while of at by for with about against between into through during before after above below to from up down in out on off over under again further then once here there when where why how all any both each few more most other some such no nor not only own same so than too very s t can will just don don't should should've now d ll m o re ve y ain aren aren't couldn couldn't didn didn't doesn doesn't hadn hadn't hasn hasn't haven haven't isn isn't ma mightn mightn't mustn mustn't needn needn't shan shan't shouldn shouldn't wasn wasn't weren weren't won won't wouldn wouldn't""".split()


def load_stopwords():
    words = set(NLTK_STOPWORDS)
    words.update({"im", "ive", "id", "ill", "dont", "didnt", "doesnt", "cant", "couldnt", "wouldnt"})
    return words


STOP_WORDS = load_stopwords()


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return " ".join(w for w in text.split() if w not in STOP_WORDS)


vectorizer = joblib.load("models/tfidf.pkl")
model = joblib.load("models/logreg.pkl")


def predict(text):
    features = vectorizer.transform([clean_text(text)])
    probabilities = model.predict_proba(features)[0]
    ranked = sorted(zip(model.classes_, probabilities), key=lambda p: p[1], reverse=True)
    return ranked


def show(text):
    ranked = predict(text)
    label, prob = ranked[0]
    print(f"\n  \"{text}\"\n  → {EMOTIONS[label]}  ({prob:.0%})")
    for other, p in ranked[1:3]:
        print(f"     {EMOTIONS[other]:<14} {p:.0%}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        show(" ".join(sys.argv[1:]))
    else:
        print("Type a message (empty line to quit).")
        while True:
            line = input("\n> ").strip()
            if not line:
                break
            show(line)

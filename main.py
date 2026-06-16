import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from neural_network import train_nn
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

def load_data():

    df = pd.read_csv("data/processed/emotion_clean.csv")

    df = df.dropna(subset=["clean_text"])
    df = df[df["clean_text"].str.strip() != ""]

    return df

def preprocess_data(df):

    vectorizer = TfidfVectorizer(max_features=5000)

    X = vectorizer.fit_transform(df["clean_text"])
    y = df["label"]

    return X, y, vectorizer

def evaluate_model(name, y_test, predictions):

    print(f"\n{name}")

    print("Accuracy:", accuracy_score(y_test, predictions))
    print("Precision:", precision_score(y_test, predictions, average="macro"))
    print("Recall:", recall_score(y_test, predictions, average="macro"))
    print("F1-score:", f1_score(y_test, predictions, average="macro"))

    print(classification_report(y_test, predictions))

def main():

    df = load_data()

    X, y, vectorizer = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    lr = LogisticRegression(max_iter=5000)
    lr.fit(X_train, y_train)
    pred_lr = lr.predict(X_test)

    svm = LinearSVC()
    svm.fit(X_train, y_train)
    pred_svm = svm.predict(X_test)

    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    pred_nb = nb.predict(X_test)

   

    joblib.dump(lr, "models/logreg.pkl")
    joblib.dump(svm, "models/svm.pkl")
    joblib.dump(nb, "models/nb.pkl")
    joblib.dump(vectorizer, "models/tfidf.pkl")

    pred_nn = train_nn(X_train, X_test, y_train)

    evaluate_model("Logistic Regression", y_test, pred_lr)
    evaluate_model("SVM", y_test, pred_svm)
    evaluate_model("Naive Bayes", y_test, pred_nb)
    evaluate_model("Neural Network", y_test, pred_nn)

    models = ["LogReg", "SVM", "Naive Bayes", "NN"]

    accuracies = [
        accuracy_score(y_test, pred_lr),
        accuracy_score(y_test, pred_svm),
        accuracy_score(y_test, pred_nb),
        accuracy_score(y_test, pred_nn)
    ]

    f1_scores = [
        f1_score(y_test, pred_lr, average="macro"),
        f1_score(y_test, pred_svm, average="macro"),
        f1_score(y_test, pred_nb, average="macro"),
        f1_score(y_test, pred_nn, average="macro")
    ]

    x = np.arange(len(models))
    width = 0.35

    plt.figure(figsize=(8, 5))

    plt.bar(x - width/2, accuracies, width, label="Accuracy")
    plt.bar(x + width/2, f1_scores, width, label="F1-score")

    plt.xticks(x, models)

    plt.ylabel("Score")
    plt.title("Model Comparison")

    plt.legend()

    plt.savefig("images/model_comparison.png")

    plt.show()

    cm = confusion_matrix(y_test, pred_svm)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    
    plt.savefig("images/confusion_matrix.png")
    plt.show()

    params = {"C": [0.01, 0.1, 1, 10]}

    grid = GridSearchCV(
        LinearSVC(),
        params,
        cv=5,
        scoring="f1_macro"
    )

    grid.fit(X_train, y_train)

    print("Best params:", grid.best_params_)
    print("Best score:", grid.best_score_)

if __name__ == "__main__":
    main()
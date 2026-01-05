import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# === Ścieżki ===
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "polemo_clean.csv")
MODEL_OUT = os.path.join(ROOT, "models", "tfidf_logreg.joblib")

def main():
    if not os.path.exists(DATA):
        raise FileNotFoundError(f"Nie znaleziono pliku danych: {DATA}")

    # === Wczytanie danych ===
    df = pd.read_csv(DATA)
    df = df.dropna(subset=["text", "label"])
    X = df["text"].astype(str).tolist()
    y = df["label"].astype(str).tolist()
    print(f" Wczytano {len(df)} rekordów z pliku {DATA}")

    # === Podział na train/test ===
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # === Pipeline TF-IDF + Logistic Regression ===
    print("Trening modelu TF-IDF + Logistic Regression...")
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            max_features=30000,
            min_df=3,
            max_df=0.9
        )),
        ("clf", LogisticRegression(
            max_iter=500,
            solver="lbfgs",
            multi_class="auto",
            n_jobs=-1
        ))
    ])

    pipe.fit(X_train, y_train)

    # === Ewaluacja ===
    print("\n Wyniki na zbiorze testowym:")
    y_pred = pipe.predict(X_test)
    print(classification_report(y_test, y_pred, digits=3))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    # === Zapis modelu ===
    os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
    joblib.dump(pipe, MODEL_OUT)
    print(f"\n Model zapisano do: {MODEL_OUT}")

if __name__ == "__main__":
    main()

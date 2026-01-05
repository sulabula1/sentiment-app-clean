import os
import pandas as pd

# Folder, gdzie masz pliki np. all.sentence.train.txt
POLEMO_DIR = r"C:\Users\Jakub\Desktop\PolEmo2"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT = os.path.join(ROOT, "data", "polemo_clean.csv")

def load_file(filename):
    """Wczytuje plik w formacie: tekst ... __label__etykieta"""
    path = os.path.join(POLEMO_DIR, filename)
    print(f"Wczytuję {filename}...")
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # znajdź pozycję etykiety "__label__"
            if "__label__" in line:
                text, label = line.rsplit("__label__", 1)
                text = text.strip()
                label = label.strip()
                rows.append((text, label))
    df = pd.DataFrame(rows, columns=["text", "label"])
    print(f"   → {len(df)} rekordów")
    return df

def main():
    files = ["all.sentence.train.txt", "all.sentence.dev.txt", "all.sentence.test.txt"]
    dfs = [load_file(f) for f in files if os.path.exists(os.path.join(POLEMO_DIR, f))]
    df = pd.concat(dfs, ignore_index=True)
    print(f"\n Połączony zbiór: {len(df)} rekordów")

    # Mapowanie etykiet
    mapping = {
        "z_plus_m": "pozytywny",
        "z_minus_m": "negatywny",
        "z_zero": "neutralny"
    }
    df["label"] = df["label"].map(mapping)
    df = df.dropna(subset=["text", "label"])

    # Zapis do CSV
    os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
    df.to_csv(OUTPUT, index=False, encoding="utf-8")
    print(f" Zapisano oczyszczony zbiór do: {OUTPUT}")
    print(df.sample(5))

if __name__ == "__main__":
    main()

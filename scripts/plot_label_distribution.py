import pandas as pd
import matplotlib.pyplot as plt

# Wczytanie danych
df = pd.read_csv("data/polemo_clean.csv")

# Sprawdzenie unikalnych klas (opcjonalnie)
print(df["label"].value_counts())

# Wykres słupkowy
counts = df["label"].value_counts()

plt.figure(figsize=(6, 4))
counts.plot(kind="bar")
plt.title("Rozkład klas sentymentu w zbiorze polemo_clean.csv")
plt.xlabel("Klasa sentymentu")
plt.ylabel("Liczba przykładów")

plt.tight_layout()
plt.savefig("data/rozkład_klas.png", dpi=300)
plt.show()
# Aplikacja webowa do analizy sentymentu opinii (TF-IDF vs HerBERT)

**Projekt pracy inżynierskiej – wersja finalna.**  
Repozytorium zawiera kod źródłowy aplikacji webowej stanowiącej część praktyczną pracy inżynierskiej.  
Projekt został udostępniony w formie repozytorium GitHub ze względu na rozmiar przekraczający 600 MB oraz konieczność zachowania pełnej struktury, zależności i historii zmian. Wytrenowane modele oraz dane nie są dołączone (limity GitHub / ograniczenia licencyjne).

---

Repozytorium zawiera aplikację webową służącą do analizy sentymentu opinii w języku polskim.  
System umożliwia wybór metody klasyfikacji:
- **TF-IDF + klasyfikator** (model klasyczny, szybki),
- **HerBERT** (model transformerowy, dokładniejszy).

Aplikacja składa się z:
- **frontend** (HTML/CSS/JavaScript) – interfejs w przeglądarce,
- **backend** (FastAPI) – REST API obsługujące analizę sentymentu.

---

## Funkcje aplikacji

- wprowadzanie opinii tekstowej,
- wybór modelu (TF-IDF / HerBERT),
- wynik sentymentu + prawdopodobieństwa klas,
- komunikacja frontend ↔ backend poprzez REST API.

---

## Wymagania środowiskowe

- **Python 3.10–3.12** (zalecane)  
  > Uwaga: Python 3.14 może powodować problemy kompatybilności (np. ostrzeżenia Pydantic / zależności).
- pip (menedżer pakietów)
- system Windows/Linux/macOS (testowane pod systemem Windows, ale na innych powinno działać stabilnie)

---

## Struktura pracy (przykładowa)

sentiment_app/
├─ backend/
│ └─ main.py
├─ frontend/
│ └─ index.html
├─ scripts/ # skrypty pomocnicze (opcjonalnie)
├─ requirements.txt
├─ .gitignore
└─ README.md

---

## Instalacja i uruchomienie aplikacji

1. Klonowanie repozytorium (bash)
git clone <URL_DO_REPOZYTORIUM>
cd sentiment_app

2. Utworzenie i aktywacja środowiska wirtualnego

- **Windows (powershell)**
python -m venv venv
.\venv\Scripts\Activate.ps1
- **Linux/MacOS**
python3 -m venv venv
source venv/bin/activate

3. Instalcja zależności
pip install -r requirements.txt

---

## Uruchomienie backendu (FastAPI)

uvicorn backend.main:app --reload
> Po uruchomieniu backend będzie dostępny pod adresem:
http://127.0.0.1:8000
- Automatyczna dokumentacja API (Swagger):
http://127.0.0.1:8000/docs

---

## Uruchomienie frontedu 

Frontend aplikacji jest stroną statyczną.
Najprostszy sposób:
- otworzyć plik frontend/index.html bezpośrednio w przeglądarce.

**Alternatywnie** przy problemach z CORS (cmd):
cd frontend
python -m http.server 5500
- następnie w przeglądarce:
http://127.0.0.1:5500

---

## Modele i dane

Repozytorium zawiera kod aplikacji oraz skrypty pomocnicze, ale **nie zawiera**:
- wytrenowanych modeli (np. `.joblib`, `.safetensors`),
- zbioru PolEmo 2.0 ani jego oczyszczonej wersji (`polemo_clean.csv`).

Jest to świadoma decyzja projektowa, ponieważ:
- modele transformerowe zajmują dużo miejsca, a GitHub posiada limity rozmiaru plików,
- dane źródłowe (PolEmo 2.0) mogą podlegać ograniczeniom licencyjnym,
- duże pliki binarne utrudniają klonowanie repozytorium i pracę z historią Gita.
  
---

## Skrypty w repozytorium
W katalogu `scripts/` znajdują się:
- `prepare_polemo.py` – wstępne przygotowanie danych (konwersja/łączenie, mapowanie etykiet),
- `train_tfidf.py` – trening klasycznego modelu TF-IDF + klasyfikator i zapis artefaktu modelu,
- `train_herbert.py` – trening / fine-tuning modelu HerBERT (np. w środowisku Google Colab).

Aby odtworzyć trening modeli, należy:
1. pozyskać zbiór PolEmo 2.0 zgodnie z jego zasadami udostępniania,
2. uruchomić skrypt przygotowania danych,
3. uruchomić trening wybranego modelu.

Po wytrenowaniu modele są zapisywane lokalnie i wykorzystywane przez backend aplikacji.

---

## Test działania API

**Przykładowe zapytanie testowe dla backendu dla TF-IDF:**
>
curl -X POST "http://127.0.0.1:8000/analyze?model=tfidf" \
-H "Content-Type: application/json" \
-d "{\"text\":\"Ten produkt jest świetny!\"}"

**Przykładowe zapytanie testowe dla backendu dla HerBERT:**
>
curl -X POST "http://127.0.0.1:8000/analyze?model=herbert" \
-H "Content-Type: application/json" \
-d "{\"text\":\"Ten produkt jest świetny!\"}"

---

## Informacje końcowe 

Projekt został zrealizowany w ramach pracy inżynierskiej jako jej część praktyczna.  
Repozytorium prezentuje architekturę aplikacji webowej oraz sposób integracji klasycznych
i transformerowych metod przetwarzania języka naturalnego z technologiami webowymi.

**Oddawana wersja projektu:** branch `main` 26.01.2026.

  

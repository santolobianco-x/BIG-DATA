# Recommendation System - MovieLens

Questo progetto implementa un **sistema di raccomandazione** basato sul dataset **MovieLens 100K** e sull'algoritmo **Alternating Least Squares (ALS)** di Apache Spark MLlib.

L'obiettivo è utilizzare i rating degli utenti per:
- calcolare statistiche descrittive sui dati;
- addestrare un modello di *Collaborative Filtering* tramite ALS;
- valutare le prestazioni del modello tramite **RMSE**;
- generare la **Top 10 dei film raccomandati** per ciascun utente.

---

## Dataset

Il progetto utilizza **MovieLens 100K**, composto da 100.000 valutazioni fornite da 943 utenti su circa 1.700 film.

- **Download:** [MovieLens 100K (.zip)](https://files.grouplens.org/datasets/movielens/ml-100k.zip)

Dopo aver estratto l'archivio, posizionare nella cartella `data/` esclusivamente i seguenti file:
- `u1.base`
- `u1.test`
- `u.item`

---

## Tecnologie utilizzate

- **Python**
- **PySpark / Apache Spark MLlib** (ALS, `RegressionEvaluator`)
- **Spark DataFrame API**

---

## Struttura del progetto

```text
recommendation_system/
│
├── data/
│   ├── u1.base
│   ├── u1.test
│   └── u.item
│
├── main.py
├── preprocessing.py
# Recommendation System - MovieLens

Questo progetto implementa un **sistema di raccomandazione** basato sul dataset **MovieLens 100K** e sull'algoritmo **Alternating Least Squares (ALS)** di Apache Spark MLlib.

L'obiettivo è utilizzare i rating degli utenti per:
- calcolare statistiche descrittive sui dati;
- addestrare un modello di *Collaborative Filtering* tramite ALS;
- valutare le prestazioni del modello tramite **RMSE**;
- generare la **Top 10 dei film raccomandati** per ciascun utente.

---

## Dataset

Il progetto utilizza **MovieLens 100K**, composto da 100.000 valutazioni fornite da 943 utenti su circa 1.700 film.

- **Download:** [MovieLens 100K (.zip)](https://files.grouplens.org/datasets/movielens/ml-100k.zip)

Dopo aver estratto l'archivio, posizionare nella cartella `data/` esclusivamente i seguenti file:
- `u1.base`
- `u1.test`
- `u.item`

---

## Tecnologie utilizzate

- **Python**
- **PySpark / Apache Spark MLlib** (ALS, `RegressionEvaluator`)
- **Spark DataFrame API**

---

## Struttura del progetto

```text
recommendation_system/
│
├── data/
│   ├── u1.base
│   ├── u1.test
│   └── u.item
│
├── main.py
├── preprocessing.py
├── recommendation.py
└── recommendation.py

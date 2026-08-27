# Recommendation System - MovieLens

Questo progetto implementa un **sistema di raccomandazione** basato sul dataset **MovieLens 100K** e sull'algoritmo **Alternating Least Squares (ALS)** di Apache Spark MLlib.

Il progetto permette di:

- calcolare statistiche sui rating;
- addestrare un modello di *Collaborative Filtering* tramite ALS;
- valutare il modello tramite **RMSE**;
- generare la **Top 10 delle raccomandazioni** per ciascun utente.

---

## Dataset

Il progetto utilizza **MovieLens 100K**, composto da 100.000 valutazioni effettuate da 943 utenti su circa 1.700 film.

**Download:** [MovieLens 100K](https://files.grouplens.org/datasets/movielens/ml-100k.zip)

Dopo aver estratto l'archivio, mantenere nella cartella `data/` solamente:

- `u1.base` — training set;
- `u1.test` — test set;
- `u.item` — informazioni sui film e relativi titoli.

---

## Tecnologie utilizzate

- **Python**
- **PySpark / Apache Spark MLlib**
- **ALS** per il Collaborative Filtering
- **Spark DataFrame API**
- **RegressionEvaluator** per il calcolo del RMSE

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
└── README.md
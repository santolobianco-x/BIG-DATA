# Recommendation System - MovieLens

Questo progetto implementa un semplice **sistema di raccomandazione** utilizzando il dataset **MovieLens 100K** e l'algoritmo **Alternating Least Squares (ALS)** disponibile in Apache Spark MLlib.

L'obiettivo è utilizzare le valutazioni degli utenti sui film per:
- calcolare alcune statistiche sui dati;
- addestrare un modello di Collaborative Filtering tramite ALS;
- valutare il modello utilizzando il **RMSE**;
- generare le **10 migliori raccomandazioni per ogni utente**.

---

## Dataset

Il progetto utilizza il dataset **MovieLens 100K**, che contiene circa 100.000 valutazioni effettuate da 943 utenti su circa 1.700 film.


L'obiettivo è utilizzare le valutazioni degli utenti sui film per:

- calcolare alcune statistiche sui dati;
- addestrare un modello di Collaborative Filtering tramite ALS;
- valutare il modello utilizzando il **RMSE**;
- generare le **10 migliori raccomandazioni per ogni utente**.

---

## Tecnologie utilizzate

Il progetto è sviluppato in **Python** e utilizza:

- **PySpark** per l'elaborazione distribuita dei dati;
- **Apache Spark MLlib** per l'algoritmo di Collaborative Filtering ALS;
- **Spark DataFrame** per la gestione e l'analisi dei dati;
- **RegressionEvaluator** di Spark MLlib per il calcolo del RMSE.

---

Il dataset può essere scaricato dal sito ufficiale GroupLens:

https://files.grouplens.org/datasets/movielens/ml-100k.zip

Dopo aver scaricato ed estratto l'archivio, nella cartella `data/` devono essere mantenuti solamente i seguenti file:

```text
data/
├── u1.base
├── u1.test
└── u.item

---

## Struttura del progetto

La struttura del progetto è la seguente:

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

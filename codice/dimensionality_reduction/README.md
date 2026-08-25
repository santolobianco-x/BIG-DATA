# Big Data - Dimensionality Reduction

Il progetto implementa alcuni algoritmi di riduzione della dimensionalità,
clustering e fattorizzazione di matrici utilizzando principalmente Apache Spark.

## Dataset

Per PCA, KMeans e CUR viene utilizzato il dataset **Breast Cancer** disponibile
nella libreria `scikit-learn`.

## Struttura

- `preprocessing.py`  
  Carica il dataset Breast Cancer e prepara i dati convertendoli in vettori
  Spark (`RDD`).

- `pca.py`  
  Applica la **PCA** tramite `pyspark.mllib` e permette di visualizzare le
  prime due componenti principali.

- `clustering.py`  
  Applica **KMeans con k=2** sia ai dati originali sia ai dati trasformati
  dalla PCA. Calcola inoltre la **Silhouette** per confrontare la qualità
  dei due clustering.

- `cur.py`  
  Implementa la **CUR Decomposition**, selezionando righe e colonne della
  matrice in base alla loro norma e calcolando la matrice centrale tramite
  pseudoinversa.

- `main.py`  
  Coordina il caricamento dei dati e l'esecuzione di PCA, KMeans e CUR.

- `nmf.py`  
  Mostra un approccio alla **NMF tramite ALS**, utilizzando dati sparsi di
  esempio e il vincolo di non negatività. Vengono visualizzati i fattori
  latenti ottenuti per utenti ed elementi.

## Tecnologie

- Python
- Apache Spark / PySpark
- Spark MLlib
- NumPy
- Pandas
- Matplotlib
- Scikit-learn


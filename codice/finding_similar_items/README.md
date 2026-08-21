# Finding Similar Items

Questo progetto realizza un sistema di **classificazione di recensioni Amazon basato sulla similarità tra documenti**, utilizzando **MinHash e Locality Sensitive Hashing (LSH)**.

## Dataset

Il dataset utilizzato è **Amazon Reviews**, disponibile su Kaggle:

[Amazon Reviews – Kaggle](https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews?resource=download&utm_source=chatgpt.com)

Il dataset contiene recensioni Amazon associate a una valutazione di popolarità.

## Funzionamento

Il programma:

1. carica i dati di training e test;
2. combina titolo e testo delle recensioni;
3. suddivide ogni recensione in **shingle** di 3 caratteri;
4. costruisce un vocabolario degli shingle e rappresenta ogni documento tramite un **vettore binario sparso**;
5. applica **MinHash LSH** per individuare rapidamente le recensioni più simili;
6. utilizza i **3 vicini più prossimi (k-NN)** per predire la classe della recensione;
7. confronta le predizioni con i valori reali e calcola l'**accuracy**.

Il programma utilizza **Apache Spark/PySpark** per la gestione e l'elaborazione dei dati.


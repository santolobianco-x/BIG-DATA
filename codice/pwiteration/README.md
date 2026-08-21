# Power Iteration con PySpark

## Descrizione

Questo progetto implementa l'algoritmo **Power Iteration** utilizzando **PySpark** per calcolare l'**autovettore dominante** di una matrice.

La Power Iteration è un metodo iterativo utilizzato per trovare l'autovettore associato all'autovalore dominante di una matrice. L'algoritmo parte da un vettore iniziale e, ad ogni iterazione, calcola il prodotto tra la matrice e il vettore, normalizzando successivamente il risultato:

$$
v_{k+1} = \frac{A v_k}{\lVert A v_k \rVert}
$$

Il procedimento viene ripetuto fino al raggiungimento della convergenza oppure fino al numero massimo di iterazioni.

## Implementazione

Il progetto è suddiviso in tre file:

- **`main.py`**: contiene la matrice e il vettore iniziale, crea il contesto Spark e avvia l'algoritmo.
- **`poweriteration.py`**: contiene l'implementazione principale della Power Iteration. Ad ogni iterazione calcola il prodotto matrice-vettore, normalizza il risultato e verifica la convergenza confrontando il nuovo vettore con quello precedente.
- **`matrix_vector.py`**: contiene la funzione per il prodotto matrice-vettore distribuito tramite Spark.

La matrice viene rappresentata in formato sparso tramite tuple:

```text
(riga, colonna, valore)

# PageRank e confronto tra modelli di grafi

## Descrizione del progetto

Il progetto analizza una rete rappresentata tramite un **grafo diretto** e studia il comportamento dell'algoritmo **PageRank**.

L'obiettivo principale è analizzare la struttura della rete originale e confrontare il vettore PageRank ottenuto dalla rete reale con quello ottenuto su due differenti modelli di grafi generati artificialmente:

* **Random Graph**, cioè un grafo casuale;
* **Barabási-Albert Graph**, cioè un grafo generato secondo il principio della *preferential attachment*.

Il confronto finale viene effettuato tramite la **cosine similarity** tra i vettori PageRank.

Il progetto è stato organizzato in più file Python in modo da separare:

* le operazioni di analisi del grafo;
* la generazione dei modelli di grafo;
* l'esecuzione complessiva dell'esercizio.

---

# Obiettivo dell'esercizio

Dato un dataset contenente gli archi di una rete Web, il progetto esegue le seguenti operazioni:

1. costruzione del grafo diretto a partire dal dataset;
2. analisi della distribuzione di `in-degree` e `out-degree`;
3. visualizzazione delle distribuzioni in scala log-log;
4. approssimazione delle distribuzioni tramite una **power law**;
5. individuazione della componente debolmente connessa più grande;
6. utilizzo di tale componente per le operazioni successive;
7. calcolo del PageRank del grafo originale;
8. individuazione dei nodi con PageRank più elevato;
9. generazione di un Random Graph con un numero di nodi pari a quello della componente originale;
10. generazione di un grafo Barabási-Albert con lo stesso numero di nodi e con un numero di archi almeno pari a quello del grafo originale;
11. calcolo del PageRank dei due grafi generati;
12. ordinamento dei rispettivi vettori PageRank;
13. confronto tra il grafo originale e i due modelli tramite cosine similarity.

---

# Struttura del progetto

```text
pagerank/
│
├── data/
│   └── unict.txt
│
├── main.py
├── graph_utils.py
├── graph_models.py
└── README.md
```

## Descrizione dei file

### `data/unict.txt`

È il dataset utilizzato dal progetto.

Ogni riga rappresenta un collegamento tra due nodi della rete.

Il file viene letto utilizzando:

```python
nx.read_edgelist()
```

e successivamente trasformato in un grafo diretto.

---

### `main.py`

È il **programma principale**.

Non contiene direttamente l'implementazione delle operazioni più specifiche, ma richiama le funzioni definite nei moduli:

* `graph_utils.py`
* `graph_models.py`

Il suo compito è quindi quello di stabilire **l'ordine con cui vengono eseguite le varie operazioni dell'esercizio**.

---

### `graph_utils.py`

Contiene le funzioni di utilità e di analisi del grafo.

In particolare contiene funzioni per:

* caricare il grafo;
* stampare le informazioni;
* ottenere gli `in-degree`;
* ottenere gli `out-degree`;
* visualizzare e approssimare le distribuzioni;
* trovare la componente più grande;
* calcolare il PageRank;
* calcolare la cosine similarity.

---

### `graph_models.py`

Contiene le funzioni relative alla **generazione dei modelli di grafo**.

In particolare:

* generazione del Random Graph;
* ricerca del parametro `m` per il modello Barabási-Albert;
* generazione del grafo Barabási-Albert.

---

# Librerie utilizzate

Il progetto utilizza le seguenti librerie Python.

## NetworkX
## NumPy
## Matplotlib
## SciPy

---


# Organizzazione del codice

Il progetto è stato suddiviso in moduli per mantenere separati i diversi compiti.

## `graph_utils.py`

Contiene le operazioni di analisi:

```text
load_graph()
print_graph_info()
get_in_degree()
get_out_degree()
power_law()
plot_and_fit()
get_largest_component()
get_pagerank()
cosine_similarity()
```

Queste funzioni non si occupano di creare nuovi modelli di grafo, ma di **caricare, analizzare e confrontare grafi**.

---

## `graph_models.py`

Contiene le operazioni relative ai modelli:

```text
generate_random_graph()
find_ba_m()
generate_ba_graph()
```

Queste funzioni si occupano quindi della **generazione dei grafi artificiali**.

---

## `main.py`

Contiene il flusso principale dell'esercizio:

```text
Caricamento
     ↓
Analisi degree
     ↓
Componente più grande
     ↓
PageRank originale
     ↓
Random Graph
     ↓
Barabási-Albert
     ↓
PageRank dei modelli
     ↓
Ordinamento vettori
     ↓
Cosine similarity
     ↓
Confronto finale
```

In questo modo `main.py` rimane principalmente responsabile dell'**esecuzione e del coordinamento** delle varie operazioni, mentre i dettagli implementativi sono contenuti nei moduli separati.

---

# Come eseguire il progetto

Per eseguire il progetto è necessario avere installato Python e le librerie utilizzate.

Le principali dipendenze sono:

```bash
pip install networkx numpy matplotlib scipy
```

Una volta posizionati nella directory principale del progetto, è sufficiente eseguire:

```bash
python main.py
```

Il programma caricherà automaticamente il dataset:

```text
data/unict.txt
```

e procederà con tutte le operazioni dell'esercizio.

---


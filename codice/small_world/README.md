# Navigabilità nelle reti Small World

## Descrizione del progetto

Il progetto studia la **navigabilità delle reti Small World** attraverso un algoritmo di **greedy routing**.

La rete viene rappresentata tramite una **griglia bidimensionale `n × n`**, nella quale ogni nodo è collegato ai propri vicini locali e può possedere anche alcuni **collegamenti a lunga distanza** (*long links*).

L'obiettivo principale è confrontare il comportamento del greedy routing su due differenti modelli di rete:

* **Watts-Strogatz**, in cui i collegamenti a lunga distanza vengono scelti casualmente;

* **Kleinberg**, in cui la probabilità di scegliere un collegamento a lunga distanza dipende dalla distanza tra i nodi.

Il routing viene effettuato utilizzando la **distanza di Manhattan** e, ad ogni passo, viene scelto il vicino che permette di avvicinarsi maggiormente alla destinazione.

Il progetto analizza inoltre come varia il numero medio di passi necessari per raggiungere la destinazione al crescere della dimensione della rete e studia, come esperimento bonus, l'effetto del parametro `alpha` nel modello Kleinberg.

---

# Obiettivo dell'esercizio

Dato un insieme di nodi organizzati su una griglia `n × n`, il progetto esegue le seguenti operazioni:

1. costruzione di una rete a griglia;

2. definizione dei collegamenti locali tra nodi adiacenti;

3. aggiunta di collegamenti a lunga distanza secondo il modello **Watts-Strogatz**;

4. aggiunta di collegamenti a lunga distanza secondo il modello **Kleinberg**;

5. definizione della distanza di Manhattan tra i nodi;

6. implementazione dell'algoritmo di **greedy routing**;

7. scelta casuale di una coppia di nodi `source` e `target`;

8. esecuzione del routing dalla sorgente alla destinazione;

9. misurazione del numero di passi necessari per raggiungere il target;

10. ripetizione dell'esperimento per più coppie di nodi;

11. calcolo del numero medio di passi e del tasso di successo;

12. ripetizione degli esperimenti per diverse dimensioni della griglia;

13. confronto sperimentale tra Watts-Strogatz e Kleinberg;

14. confronto con le curve teoriche di scaling;

15. variazione del parametro `alpha` nel modello Kleinberg.

---

# Struttura del progetto

```text
small_world/

│
├── main.py
├── grid_network.py
├── graph_models.py
├── routing.py
├── experiments.py
├── plots.py
├── small_world_scaling.png
├── alpha_sweep.png
└── README.md
```

## Descrizione dei file

### `grid_network.py`

Contiene la classe:

```python
GridNetwork
```

che rappresenta la rete come una griglia `n × n`.

La classe gestisce:

* i nodi della rete;

* i vicini locali;

* i collegamenti a lunga distanza;

* la distanza di Manhattan;

* l'insieme complessivo dei vicini di un nodo.

---

### `graph_models.py`

Contiene le funzioni utilizzate per costruire i due modelli di rete:

* `build_watts_strogatz()`;

* `build_kleinberg()`.

Le due funzioni partono dalla stessa struttura di base, cioè una griglia `n × n`, ma utilizzano strategie differenti per scegliere i collegamenti a lunga distanza.

---

### `routing.py`

Contiene l'algoritmo di **greedy routing**:

```python
greedy_route()
```

La funzione riceve:

* la rete;

* il nodo sorgente;

* il nodo destinazione.

Ad ogni passo considera tutti i vicini del nodo corrente e sceglie quello con distanza di Manhattan minima rispetto alla destinazione.

---

### `experiments.py`

Contiene le funzioni utilizzate per effettuare gli esperimenti:

```text
measure_routing_performance()

scaling_experiment()
```

La prima misura le prestazioni del routing su una singola dimensione della rete.

La seconda ripete l'esperimento per diverse dimensioni della griglia e per entrambi i modelli.

---

### `plots.py`

Contiene le funzioni per la produzione dei grafici:

```text
plot_scaling()

plot_alpha_sweep()
```

La prima visualizza lo scaling del numero medio di passi.

La seconda visualizza l'effetto della variazione di `alpha` nel modello Kleinberg.

---

### `main.py`

È il **programma principale**.

Si occupa di coordinare tutte le operazioni dell'esercizio:

* costruzione delle reti;

* test del greedy routing;

* esecuzione degli esperimenti;

* stampa dei risultati;

* generazione dei grafici.

---

# Librerie utilizzate

Il progetto utilizza le seguenti librerie Python.

## `random`

Utilizzata per la generazione casuale dei nodi e dei collegamenti nel modello Watts-Strogatz e per la scelta casuale delle coppie `source-target`.

## NumPy

Utilizzata principalmente per:

* generazione delle probabilità nel modello Kleinberg;

* scelta casuale secondo una distribuzione di probabilità;

* calcolo di media e mediana dei risultati.

## Matplotlib

Utilizzata per la visualizzazione dei risultati sperimentali attraverso i grafici.

---

# Organizzazione del codice

Il progetto è stato suddiviso in moduli per mantenere separati i diversi compiti.

## `grid_network.py`

Contiene le operazioni relative alla rappresentazione della rete:

```text
GridNetwork()

neighbors_local()

manhattan()

all_neighbors()

add_long_link()
```

Queste funzioni permettono di creare e manipolare la griglia e i suoi collegamenti.

---

## `graph_models.py`

Contiene le operazioni relative alla costruzione dei modelli:

```text
build_watts_strogatz()

build_kleinberg()
```

Le due funzioni differiscono nel modo in cui vengono scelti i collegamenti a lunga distanza.

### Watts-Strogatz

I long links vengono scelti casualmente.

In modo semplificato:

$$
P(u,v)=\text{costante}
$$

per i possibili nodi `v`.

### Kleinberg

La probabilità dipende dalla distanza:

$$
P(u,v)\propto d(u,v)^{-\alpha}
$$

dove `d(u,v)` è la distanza di Manhattan.

---

## `routing.py`

Contiene:

```text
greedy_route()
```

L'algoritmo segue una strategia locale.

A ogni passo:

```text
nodo corrente
      ↓
considera tutti i vicini
      ↓
calcola la distanza dal target
      ↓
sceglie il vicino più vicino
      ↓
ripete
```

Il routing termina quando:

* viene raggiunta la destinazione;

* non esistono vicini;

* nessun vicino permette di avvicinarsi alla destinazione;

* viene raggiunto il numero massimo di passi consentiti.

La funzione restituisce:

```text
path
steps
reached
optimal
```

dove:

* `path` è il percorso seguito;

* `steps` è il numero di passi effettuati;

* `reached` indica se la destinazione è stata raggiunta;

* `optimal` rappresenta la distanza Manhattan iniziale tra sorgente e destinazione.

---

# Distanza di Manhattan

La distanza utilizzata dal greedy routing è la **distanza di Manhattan**.

La distanza tra due nodi:

$$
a=(i_1,j_1)
$$

e:

$$
b=(i_2,j_2)
$$

è definita come:

$$
d(a,b)=|i_1-i_2|+|j_1-j_2|
$$

Ad esempio:

```text
A = (0,0)
B = (3,5)
```

ha distanza:

$$
d(A,B)=3+5=8
$$

La distanza di Manhattan rappresenta quindi il numero minimo di movimenti locali necessari per raggiungere un nodo da un altro sulla griglia.

---

# Greedy Routing

Il **greedy routing** è una strategia di instradamento che, ad ogni passo, sceglie il vicino che appare più vicino alla destinazione.

Supponendo di avere:

```text
source = (0,0)
target = (9,9)
```

l'algoritmo parte da `(0,0)` e considera tutti i suoi vicini.

Tra questi sceglie quello che minimizza:

$$
d(v,target)
$$

dove `v` è un vicino del nodo corrente.

Il procedimento viene ripetuto fino al raggiungimento del target.

Un aspetto importante è che il greedy routing utilizza solamente **informazioni locali**: non conosce necessariamente il percorso globale migliore.

---

# Modello Watts-Strogatz

Nel modello utilizzato nell'esercizio, ogni nodo della griglia riceve un certo numero di collegamenti a lunga distanza scelti casualmente.

Il parametro:

```text
num_links_per_node
```

determina il numero di long links associati a ciascun nodo.

Ad esempio:

```python
build_watts_strogatz(
    n=10,
    num_links_per_node=1
)
```

costruisce una griglia `10 × 10` e aggiunge un collegamento a lunga distanza per ogni nodo.

La scelta della destinazione del long link non dipende dalla distanza dal nodo sorgente.

---

# Modello Kleinberg

Nel modello Kleinberg i collegamenti a lunga distanza vengono scelti utilizzando una distribuzione dipendente dalla distanza.

La probabilità è:

$$
P(u,v)\propto d(u,v)^{-\alpha}
$$

dove:

* `u` è il nodo sorgente;

* `v` è il nodo destinazione;

* `d(u,v)` è la distanza di Manhattan;

* `alpha` controlla quanto la distanza influenza la probabilità.

Per:

```text
alpha = 0
```

i nodi hanno probabilità uniforme.

Aumentando `alpha`, la probabilità si concentra maggiormente sui nodi vicini.

Nel progetto viene utilizzato principalmente:

```text
alpha = 2
```

---

# Misurazione delle prestazioni

La funzione:

```text
measure_routing_performance()
```

permette di misurare le prestazioni del greedy routing.

Per ogni esperimento:

1. viene costruita una rete;

2. viene scelta casualmente una sorgente;

3. viene scelta casualmente una destinazione diversa dalla sorgente;

4. viene eseguito il greedy routing;

5. viene registrato il numero di passi;

6. viene registrato se il target è stato raggiunto;

7. l'esperimento viene ripetuto per il numero di trial specificato.

Al termine vengono calcolate diverse statistiche:

```text
steps_mean
steps_median
steps_all
reach_rate
optimal_mean
```

---

## `steps_mean`

Rappresenta il numero medio di passi necessari al greedy routing.

È la principale misura utilizzata per confrontare i due modelli.

---

## `steps_median`

Rappresenta la mediana del numero di passi.

È utile per avere una misura meno influenzata da eventuali valori molto elevati.

---

## `reach_rate`

Rappresenta la percentuale di routing che riescono effettivamente a raggiungere la destinazione:

$$
reach\_rate=
\frac{\text{routing riusciti}}
{\text{numero totale di trial}}
$$

---

## `optimal_mean`

Rappresenta la distanza Manhattan media tra le coppie di nodi `source-target`.

---

# Esperimento di scaling

La funzione:

```text
scaling_experiment()
```

studia come varia il numero medio di passi al crescere della dimensione della rete.

Nel progetto vengono utilizzati:

```python
n_values = [
    5, 8, 10, 12,
    15, 18, 20,
    25, 30
]
```

Poiché la rete è una griglia `n × n`, il numero totale di nodi è:

$$
N=n^2
$$

Ad esempio:

```text
n = 5   → 25 nodi

n = 10  → 100 nodi

n = 20  → 400 nodi

n = 30  → 900 nodi
```

Per ogni valore di `n` vengono eseguiti diversi trial sia sul modello Watts-Strogatz sia sul modello Kleinberg.

---

# Confronto tra i modelli

I risultati vengono memorizzati in:

```python
results = {
    "ws": [],
    "kl": []
}
```

dove:

* `ws` contiene i risultati di Watts-Strogatz;

* `kl` contiene i risultati di Kleinberg.

Successivamente vengono estratti i valori di `steps_mean` per confrontare i due modelli.

Il confronto permette di osservare come la diversa struttura dei long links influenzi la navigabilità della rete.

---

# Scaling teorico

Nel grafico vengono visualizzate anche due curve teoriche.

Per Watts-Strogatz viene utilizzata:

$$
n^{2/3}
$$

mentre per Kleinberg viene utilizzata:

$$
\log^2(n)
$$

Nel codice:

```python
ws_theory = n ** (2 / 3)

kl_theory = np.log2(n) ** 2
```

Queste curve vengono utilizzate come riferimento per confrontare l'andamento sperimentale del numero medio di passi.

---

# Grafico dello scaling

La funzione:

```text
plot_scaling()
```

genera un grafico in scala logaritmica utilizzando:

```python
plt.loglog()
```

Nel grafico vengono visualizzate:

* curva sperimentale Watts-Strogatz;

* curva sperimentale Kleinberg;

* curva teorica `n^(2/3)`;

* curva teorica `log²(n)`.

Le curve teoriche vengono riscalate verticalmente per renderle confrontabili con i dati sperimentali, senza modificarne la forma.

Il grafico viene salvato nel file:

```text
small_world_scaling.png
```

---

# Esperimento bonus: variazione di `alpha`

Il progetto contiene anche un esperimento bonus che studia il comportamento del modello Kleinberg al variare di `alpha`.

La funzione:

```text
plot_alpha_sweep()
```

considera valori di `alpha` compresi tra `0` e `4`.

Per ogni valore viene costruita una rete Kleinberg ed eseguito un insieme di trial.

Viene quindi calcolato il numero medio di passi del greedy routing.

Il risultato viene rappresentato tramite un grafico che mostra:

```text
alpha
  ↓
numero medio di passi
```

Il valore:

```text
alpha = 2
```

viene evidenziato nel grafico.

Il risultato viene salvato nel file:

```text
alpha_sweep.png
```

---

# Flusso principale dell'esercizio

Il file `main.py` segue il seguente flusso:

```text
Costruzione delle reti
          ↓
     Test rapido
          ↓
Scelta delle dimensioni n
          ↓
Esperimento Watts-Strogatz
          ↓
Esperimento Kleinberg
          ↓
Calcolo dei passi medi
          ↓
Confronto dei risultati
          ↓
Grafico dello scaling
          ↓
Variazione di alpha
          ↓
Grafico alpha sweep
```

---

# Test rapido

Prima dell'esperimento completo, `main.py` esegue un test su una griglia:

```text
10 × 10
```

Vengono costruiti:

```python
net_ws = build_watts_strogatz(
    10,
    seed=0
)

net_kl = build_kleinberg(
    10,
    alpha=2,
    seed=0
)
```

e viene effettuato il routing tra:

```text
source = (0,0)
target = (9,9)
```

In questo modo è possibile verificare rapidamente il funzionamento dell'implementazione.

---

# Esecuzione del progetto

Per eseguire il progetto è necessario avere installato Python e le librerie utilizzate.

Le principali dipendenze sono:

```bash
pip install numpy matplotlib
```

Una volta posizionati nella directory principale del progetto, è sufficiente eseguire:

```bash
python main.py
```

Il programma eseguirà automaticamente:

1. il test rapido;

2. l'esperimento di scaling;

3. la stampa dei risultati;

4. la generazione del grafico `small_world_scaling.png`;

5. l'esperimento bonus sulla variazione di `alpha`;

6. la generazione del grafico `alpha_sweep.png`.

---

# Risultato finale

Il progetto permette quindi di studiare sperimentalmente la **navigabilità delle reti Small World**.

In particolare, vengono messi in relazione:

* struttura della rete;

* distribuzione dei collegamenti a lunga distanza;

* distanza tra i nodi;

* algoritmo di greedy routing;

* numero di passi necessari;

* dimensione della rete;

* parametro `alpha`.

Il confronto tra Watts-Strogatz e Kleinberg permette di osservare come una diversa distribuzione dei collegamenti a lunga distanza possa modificare il comportamento di un algoritmo di routing che utilizza solamente informazioni locali.


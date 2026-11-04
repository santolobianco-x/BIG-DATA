# 🔗 Link Prediction in una rete sociale con GNN

## Descrizione del progetto

Questo repository contiene il codice sorgente del progetto per il corso di **Big Data (a.a. 2025/2026)**, dedicato alla *Link Prediction* in reti sociali e di co-autorialità.

L'obiettivo è valutare la capacità di diversi metodi di prevedere archi mancanti in un grafo, confrontando euristiche topologiche tradizionali con un approccio basato su Graph Neural Networks (GNN).

La soluzione implementa un encoder **GraphSAGE**, associato a due decoder alternativi: **MLP** e **Dot Product**. Le prestazioni vengono valutate attraverso metriche di classificazione e di ranking, con controlli dedicati alla prevenzione del data leakage e alla riproducibilità degli esperimenti.

## Obiettivi e funzionalità

Il progetto comprende le seguenti componenti:

* **Baseline euristiche:** Common Neighbors, Jaccard, Adamic-Adar e Preferential Attachment, implementate attraverso NetworkX.

* **Modello GNN:** encoder GraphSAGE sviluppato con PyTorch Geometric, confrontato utilizzando decoder MLP e Dot Product.

* **Metriche di valutazione:** ROC-AUC, Average Precision (AP) e Hits@K.

* **Data splitting e negative sampling:** separazione degli archi per training, validation e test, con generazione di coppie negative.

* **Controlli anti-leakage:** verifiche sulla separazione degli archi e sulla provenienza delle feature utilizzate dalla GNN.

* **Feature ablation:** confronto tra configurazioni basate su feature complete, grado, clustering e feature costanti.

* **Degree bias analysis:** analisi delle prestazioni delle baseline in funzione dei quartili di grado dei nodi.

* **Esperimenti multi-seed:** ripetizione dell'addestramento della GNN con seed differenti e calcolo di media e deviazione standard delle metriche.

* **Esportazione dei risultati:** salvataggio automatico delle metriche sperimentali in file CSV e della distribuzione dei gradi in formato immagine.

* **Analisi e Reportistica:** un Jupyter Notebook dedicato all'analisi esplorativa e alla visualizzazione comparativa dei risultati, e una relazione tecnica finale in formato PDF e Word.

## Struttura del repository

```text
.
├── data/
│   ├── ca-AstroPh.txt.gz
│   ├── ca-GrQc.txt.gz
│   └── facebook_combined.txt.gz
├── src/
│   ├── baseline.py
│   ├── data_loader.py
│   ├── evaluation.py
│   ├── graph_utils.py
│   ├── negative_sampling.py
│   ├── pipeline_data.py
│   ├── reporting.py
│   ├── reproducibility.py
│   ├── split.py
│   ├── gnn/
│   │   ├── data.py
│   │   ├── decoder.py
│   │   ├── model.py
│   │   ├── prediction.py
│   │   └── training.py
│   └── utils/
│       ├── pipeline_checks.py
│       └── sanity_checks.py
├── experiments/
│   ├── baseline_evaluation.py
│   ├── baseline_setup.py
│   ├── common.py
│   ├── export_results.py
│   ├── run_ablation.py
│   ├── run_baselines.py
│   ├── run_degree_bias.py
│   └── run_gnn.py
├── config.py
├── main.py
├── environment.yml
├── notebook.ipynb
├── RELAZIONE PROGETTO.docx
├── RELAZIONE PROGETTO.pdf
└── README.md
```

La struttura è suddivisa nelle seguenti aree principali:

* `src/`: implementazione delle funzionalità principali, tra cui caricamento dei dati, preparazione del grafo, valutazione, modello GNN e controlli.

* `experiments/`: orchestrazione degli esperimenti, esecuzione delle baseline, ablation study, analisi del degree bias ed esportazione dei risultati.

* `main.py` e `config.py`: punto di ingresso della pipeline e configurazione dei parametri globali.

* `notebook.ipynb`: notebook Jupyter per l'esplorazione interattiva del dataset e la visualizzazione critica dei risultati.

* `RELAZIONE PROGETTO`: relazione tecnica finale che descrive l'architettura, gli esperimenti e le conclusioni raggiunte.

*(Nota: La cartella `results/` non è tracciata nel repository. Verrà creata automaticamente in locale durante l'esecuzione degli esperimenti per ospitare i file CSV e le immagini generati).*

## Requisiti

Il progetto utilizza Python e un ambiente Conda configurato tramite `environment.yml`.

Le principali librerie utilizzate sono:

* PyTorch
* PyTorch Geometric
* NetworkX
* NumPy
* pandas
* scikit-learn
* Matplotlib

Le versioni e le dipendenze effettive devono essere quelle definite nel file `environment.yml`.

## Installazione

### 1. Clonare il repository

```bash
git clone <URL_DEL_REPOSITORY>
cd <CARTELLA_DEL_PROGETTO>
```

Sostituire i valori tra parentesi angolari con l'URL e il nome effettivi del repository.

### 2. Creare l'ambiente Conda

```bash
conda env create -f environment.yml
```

### 3. Attivare l'ambiente

```bash
conda activate bigdata
```

Se l'ambiente è già presente, non è necessario ricrearlo.

### 4. Verificare la configurazione del dataset

I percorsi e i parametri globali sono definiti in `config.py`. Prima di eseguire la pipeline, verificare che il dataset selezionato corrisponda al file disponibile nella cartella `data/`.

I dataset testati all'interno del progetto sono:

* `ca-AstroPh.txt.gz`
* `ca-GrQc.txt.gz`
* `facebook_combined.txt.gz`

Per utilizzare un dataset differente, verificare il percorso configurato e la compatibilità del formato con il caricatore del grafo.

## Esecuzione degli esperimenti

La pipeline è orchestrata da `main.py` e permette di selezionare gli esperimenti attraverso l'argomento `--run`.

### Esecuzione completa

```bash
python main.py --run all
```

Esegue sia gli esperimenti con le baseline sia quelli con la GNN.

### Esecuzione delle sole baseline

```bash
python main.py --run baselines
```

Esegue le euristiche topologiche e l'analisi del degree bias.

### Esecuzione della sola pipeline GNN

```bash
python main.py --run gnn
```

Esegue lo studio di ablazione delle feature e il confronto multi-seed tra i decoder MLP e Dot Product.

Se non viene specificata la modalità, il valore predefinito è `all`.

**Nota:** L'esecuzione completa può richiedere tempo, soprattutto durante l'addestramento della GNN. Per esplorare i risultati tramite il notebook Jupyter `notebook.ipynb`, è prima necessario aver eseguito gli esperimenti in locale affinché il codice generi i file CSV nella cartella `results/`.

## Analisi e Documentazione

L'esplorazione dei risultati, unitamente alla visualizzazione dei grafi e alle metriche, è accessibile e commentata in `notebook.ipynb`.
Tutta l'analisi critica della metodologia, la discussione sui risultati e sui limiti etici e tecnici è consultabile in maniera strutturata nei documenti `RELAZIONE PROGETTO.docx` e `RELAZIONE PROGETTO.pdf`.

## Metodologia

### Preparazione del grafo

Il grafo viene caricato tramite NetworkX e preparato per gli esperimenti. Gli archi vengono suddivisi in insiemi destinati al training, alla validation e al test. Vengono inoltre generate coppie negative per valutare la capacità dei modelli di distinguere le connessioni positive da quelle non presenti.

Per la GNN, il grafo utilizzato per il message passing viene mantenuto distinto dagli archi impiegati come supervisione, in modo da limitare il rischio di data leakage.

### Baseline euristiche

Le baseline stimano la probabilità di connessione tra due nodi utilizzando informazioni topologiche:

* **Common Neighbors:** numero di vicini condivisi.
* **Jaccard:** similarità tra gli insiemi dei vicini.
* **Adamic-Adar:** peso dei vicini condivisi in funzione del loro grado.
* **Preferential Attachment:** prodotto dei gradi dei due nodi.

Questi metodi forniscono un riferimento per valutare il contributo dell'approccio basato su GNN.

### GraphSAGE e decoder

GraphSAGE produce una rappresentazione vettoriale dei nodi a partire dal grafo di message passing e dalle feature nodali.

Le rappresentazioni vengono utilizzate da due decoder alternativi:

* **MLP:** combina le rappresentazioni dei nodi attraverso una rete neurale non lineare.
* **Dot Product:** calcola un punteggio attraverso il prodotto scalare diretto delle rappresentazioni.

Il confronto permette di valutare le differenze tra due strategie di scoring degli archi.

### Metriche

* **ROC-AUC:** misura la capacità di distinguere le coppie positive da quelle negative.
* **Average Precision (AP):** riassume la qualità della classificazione considerando la precisione rispetto al richiamo.
* **Hits@K:** valuta il posizionamento delle coppie positive rispetto ai candidati negativi, secondo il protocollo query-based implementato.

### Ablation e degree bias

Lo studio di ablazione confronta diverse configurazioni delle feature nodali per analizzare il contributo delle informazioni strutturali disponibili al modello.

L'analisi del degree bias studia come le prestazioni delle baseline varino in funzione del grado dei nodi, evidenziando le asimmetrie predittive tra nodi a basso grado e "hub" altamente connessi.

### Riproducibilità e controlli

La pipeline imposta seed per rendere gli esperimenti riproducibili e prevede controlli rigorosi sulle partizioni dei dati e sulle feature generate per la GNN.

## Risultati

Una volta avviata la pipeline, gli esperimenti esporteranno automaticamente i seguenti file CSV e immagini nella directory locale `results/`, assegnando nomi basati sul dataset configurato:

| File | Contenuto | 
| ----- | ----- | 
| `*_baseline_results.csv` | Metriche delle baseline | 
| `*_degree_bias_results.csv` | Analisi delle baseline per quartili di grado | 
| `*_feature_ablation_results.csv` | Risultati dell'ablazione delle feature | 
| `*_gnn_decoder_results.csv` | Risultati dettagliati dei singoli seed GNN | 
| `*_gnn_decoder_summary.csv` | Riepilogo delle metriche GNN con media e deviazione standard | 
| `*_degree_distribution.png` | Distribuzione dei gradi del grafo | 

La colonna `dataset` permette di identificare il dataset associato a ciascun risultato, facilitando il confronto tra esperimenti differenti.

## Limiti e considerazioni etiche

La link prediction può essere utile per individuare relazioni potenziali in reti sociali e di co-autorialità. Tuttavia, le connessioni previste da un modello sono stime probabilistiche, non prove dell'esistenza di una relazione reale.

L'applicazione di queste tecniche a reti sociali può comportare rischi per la privacy, soprattutto quando le predizioni vengono utilizzate per inferire relazioni non esplicitamente condivise. È quindi fondamentale considerare la provenienza dei dati, il consenso, le finalità dell'analisi e i possibili effetti delle predizioni sugli individui.

I risultati dipendono inoltre dalla struttura del dataset, dalla strategia di campionamento negativo, dalle feature impiegate e dal protocollo di valutazione. Le prestazioni ottenute non devono essere interpretate come una garanzia di generalizzazione automatica ad altre tipologie di reti.


# 🔗 Link Prediction in Reti Sociali con GNN

Questo repository contiene il codice sorgente per il progetto di tesi relativo al task di **Link Prediction su reti sociali e di co-autoraggio**.

L'obiettivo dello studio è prevedere la presenza di archi futuri o mancanti all'interno di un grafo parzialmente osservato. Il progetto mette a confronto le classiche euristiche topologiche con moderni approcci di Deep Learning per grafi basati su architetture **Encoder-Decoder (GraphSAGE)**.

## 🎯 Obiettivi e Requisiti Raggiunti

Il progetto soddisfa e supera le richieste della consegna, includendo le estensioni opzionali:

* **Baseline Euristiche:** Implementazione vettorializzata di *Common Neighbors*, *Jaccard*, *Adamic-Adar* e *Preferential Attachment* tramite NetworkX.
* **Deep Learning (GNN):** Sviluppo di un Encoder **GraphSAGE** (in PyTorch Geometric) abbinato a due tipologie di Decoder (*MLP* e *Dot Product*).
* **Metriche di Valutazione:** Valutazione globale tramite **AUC** e **Average Precision (AP)**, e valutazione di ranking query-based tramite **Hits@K** (con gestione pessimistica dei pareggi).
* **Data Splitting Rigoroso:** Isolamento assoluto tra *Message-Passing Graph* e *Supervision Edges* per azzerare il rischio di Data Leakage. Modulo dedicato ai *Sanity Checks* per validare matematicamente le partizioni.
* **🌟 Analisi del Degree Bias (Estensione):** Valutazione delle performance del modello stratificata in base ai quartili di grado dei nodi (da *Cold Start* a *Hubs*).
* **Studio di Ablazione sulle Feature:** Analisi dell'impatto delle informazioni nodali (*Degree*, *Clustering*, e feature costanti per forzare l'apprendimento puramente topologico).
* **Riproducibilità e Robustezza:** Esecuzione GNN automatizzata su **Multi-Seed** per il calcolo di media e deviazione standard, garantendo validità statistica.

## 📂 Struttura della Repository

Il codice è organizzato in moduli per garantire la massima manutenibilità e chiarezza:

```text
├── data/                       # Dataset (es. ca-AstroPh.txt.gz) [da scaricare]
├── results/                    # Cartella di output generata automaticamente per CSV e Plot
├── src/                        # CORE LOGIC
│   ├── baseline.py             # Euristiche topologiche
│   ├── data_loader.py          # Caricamento e pulizia grafo NetworkX
│   ├── evaluation.py           # Metriche (AUC, AP, Hits@K)
│   ├── graph_utils.py          # Statistiche ed esplorazione topologica
│   ├── negative_sampling.py    # Generazione dei distrattori
│   ├── pipeline_data.py        # Suddivisione rigorosa dei set (Train/Val/Test)
│   ├── reproducibility.py      # Gestione globale dei Seed
│   ├── split.py                # Logica di partizionamento per GNN
│   ├── gnn/                    # Modulo PyTorch / PyTorch Geometric
│   │   ├── data.py             # Feature engineering e conversione NetworkX -> PyG
│   │   ├── decoder.py          # Architetture MLP e Dot Product
│   │   ├── model.py            # Architettura GraphSAGE
│   │   ├── prediction.py       # Inferenza a batch
│   │   └── training.py         # Ciclo di addestramento, DropEdge e Early Stopping
│   └── utils/                  # Modulo per i Sanity Checks e controllo Anti-Leakage
├── experiments/                # ORCHESTRAZIONE ESPERIMENTI
│   ├── __init__.py             # Façade pattern per esporre i runner al main
│   ├── baseline_evaluation.py  # Pipeline di valutazione baseline
│   ├── export_results.py       # Salvataggio dinamico dei CSV
│   ├── run_ablation.py         # Feature Ablation Study
│   ├── run_baselines.py        # Orchestratore per Metodi Topologici
│   ├── run_degree_bias.py      # Esperimento di Degree-Bias Analysis
│   └── run_gnn.py              # Orchestratore per la pipeline Deep Learning
├── config.py                   # Parametri globali, path e iperparametri
├── main.py                     # Entry point principale della pipeline sperimentale
├── analisi_esplorativa.ipynb   # Notebook Jupyter per EDA, grafici e aggregazione risultati
└── environment.yml             # File di configurazione Conda per la riproducibilità

```

## ⚙️ Installazione e Setup

Il progetto utilizza **Conda** per la gestione dell'ambiente virtuale, assicurando la corretta installazione dei binari di PyTorch e PyTorch Geometric.

1. **Clona la repository** sul tuo computer e accedi alla directory principale.
2. **Crea l'ambiente virtuale** utilizzando il file di configurazione fornito:
```bash
conda env create -f environment.yml

```


3. **Attiva l'ambiente virtuale:**
```bash
conda activate bigdata

```


4. **Prepara il Dataset:**
Scarica il dataset (es. `ca-AstroPh.txt.gz` dalla repository SNAP di Stanford) e posizionalo all'interno della cartella `data/`. Assicurati che il percorso in `config.py` sia allineato con il nome del file.

## 🚀 Esecuzione degli Esperimenti

La pipeline è interamente governata dal file `main.py`. Grazie al modulo `argparse`, è possibile eseguire gli esperimenti in blocco o separatamente. I risultati verranno salvati automaticamente nella directory `results/`.

**Esecuzione completa (Consigliata per la riproduzione da zero):**

```bash
python main.py --run all

```

**Esecuzioni parziali (Utili per testing o estrazione mirata di dati):**

```bash
# Esegue esclusivamente le Baseline e l'analisi del Degree Bias
python main.py --run baselines

# Esegue esclusivamente l'Ablation Study e il Multi-seed Training della GNN
python main.py --run gnn

```

## 📊 Analisi dei Risultati (Jupyter Notebook)

Il progetto include il file `analisi_esplorativa.ipynb`. Una volta terminata l'esecuzione del file `main.py`, apri questo notebook per:

1. Visualizzare l'analisi topologica e la distribuzione dei gradi (EDA).
2. Caricare automaticamente i file CSV dalla cartella `results/`.
3. Generare i grafici comparativi a barre e a linee utilizzati per la relazione tecnica.

Per avviarlo:

```bash
jupyter notebook analisi_esplorativa.ipynb

```

---

*Progetto sviluppato come requisito d'esame. Per ulteriori dettagli metodologici e per la discussione critica dei risultati (incluse le implicazioni etiche legate alla privacy), si prega di fare riferimento alla Relazione Tecnica allegata.*

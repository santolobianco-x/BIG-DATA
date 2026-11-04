# Big Data - UNICT

Questa repository contiene il materiale didattico, gli appunti, le esercitazioni pratiche e il progetto finale per il corso di Big Data. Il repository è suddiviso in due sezioni principali: una parte dedicata ai fondamenti teorici e all'analisi di enormi moli di dati (strutturati e a grafo), e una robusta sezione pratica dedicata all'implementazione in Python, PySpark e all'utilizzo del Deep Learning sui grafi.

## 📚 Contenuti Teorici (Appunti e Lezioni del Corso)

I file di riassunto (`big data riassunto.pdf`/`.docx`) copre l'intero programma del corso, partendo dai concetti base dell'elaborazione distribuita fino ai moderni algoritmi di Machine Learning su grafi. I principali capitoli trattati sono:

*   **Elaborazione Distribuita:** Il paradigma Map-Reduce per l'elaborazione di grandi dataset (word count, join relazionali) ed evoluzione dei sistemi con Hadoop e Apache Spark.
*   **Ricerca di Similarità (High-Dimensional Data):** Tecniche per il confronto efficiente di documenti e dati ad alta dimensionalità tramite Shingling, Min-Hashing e Locality-Sensitive Hashing (LSH e varianti).
*   **Riduzione della Dimensionalità:** Metodi per comprimere le informazioni e isolare i concetti latenti: PCA (Principal Component Analysis), SVD (Singular Value Decomposition), CUR e NNMF. Preservazione delle distanze spaziali (Teorema di Johnson-Lindenstrauss).
*   **Sistemi di Raccomandazione:** Filtraggio collaborativo (Collaborative Filtering), Network-Based-Inference (NBI), Latent Semantic Indexing (LSI) e modelli ibridi.
*   **Link Analysis e Web Advertising:** Algoritmo PageRank, HITS (Hubs e Authorities), tecniche di contrasto al link spam (TrustRank). Algoritmi online per il Web Advertising (es. Adwords).
*   **Analisi di Grafi di Grandi Dimensioni:** Modelli di reti random e "Small World". Tecniche per valutare la coesione sociale: conteggio dei triangoli, subgraph matching, motif finding, network alignment e rilevamento di comunità (anche sovrapposte).
*   **Graph Learning (Deep Learning su Grafi):** Tecniche avanzate per la generazione di embedding topologici. Metodi basati su random walk (Node2Vec, DeepWalk) e architetture di Graph Neural Networks (GCN, GAT, GraphSAGE).
*   **Text Mining:** Analisi testuale mediante vettorizzazione TF-IDF ed Entity Annotation per il recupero delle informazioni rilevanti.

## 💻 Pratica: Esercitazioni, Codice e Progetto Finale

La parte pratica del corso è vastissima e si divide tra script Python e un progetto finale completo. 

*   **Algoritmi implementati da zero (`/codice`):** All'interno della cartella codice sono presenti implementazioni pratiche di:
    *   *Locality Sensitive Hashing (LSH)* e *MinHashing* per la ricerca di similarità.
    *   Riduzione dimensionale con *CUR*, *PCA* e *NMF*.
    *   Algoritmi su grafi: calcolo del *PageRank*, *Power Iteration* e simulazioni su reti *Small World* e grid routing.
    *   *Recommendation Systems* con fasi di preprocessing e filtraggio.
    *   Script *PySpark* per approcci distribuiti.
*   **Progetto Finale (`/project`):** *Link Prediction in una rete sociale con GNN*. Un solido progetto di Data Science (completo di relazione tecnica) che confronta baseline euristiche tradizionali (Adamic-Adar, Jaccard, ecc.) con architetture neurali su grafi (GraphSAGE con decoder MLP e Dot Product) per prevedere le collaborazioni future in reti di co-autorialità (AstroPh, GrQc) e social network (Facebook).

## 🛠 Tecnologie Utilizzate:
*   **Linguaggi:** Python.
*   **Elaborazione Big Data:** Apache Spark (PySpark), paradigma MapReduce.
*   **Machine Learning e Deep Learning:** PyTorch, PyTorch Geometric (PyG), scikit-learn.
*   **Elaborazione Dati e Grafi:** NetworkX, NumPy, Pandas.
*   **Ambienti:** Jupyter Notebook.
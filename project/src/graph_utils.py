"""
MODULO PER L'ANALISI TOPOLOGICA E LA VISUALIZZAZIONE DEL GRAFO
"""

import networkx as nx
import matplotlib.pyplot as plt
from config import RESULTS_DIR, DATASET_NAME

def graph_statistics(G:nx.Graph):
    return{
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "density": nx.density(G),
        "connected_components": nx.number_connected_components(G)
    }



def degree_statistics(G: nx.Graph):
    degrees = [degree for _, degree in G.degree()]

    return{
        "min_degree": min(degrees),
        "max_degree": max(degrees),
        "average_degree": sum(degrees) / len(degrees),
        "median_degree": sorted(degrees)[len(degrees)//2]
    }





def plot_degree_distribution(G: nx.Graph, title, show=False, save=True):
    """
    Genera e salva un istogramma della distribuzione dei gradi del grafo.
    Permette di identificare se la rete segue una power-law
    --> una massa enorme di nodi ha pochissime connessioni
    --> una frazione di nodi posside una quantità spropositata di connessioni
    """
    degrees = [degree for _, degree in G.degree()]

    plt.figure(figsize=(8, 5))
    plt.hist(degrees, bins=50)
    plt.xlabel("Degree")
    plt.ylabel("Number of nodes")
    plt.title(title)

    if save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        output_path = RESULTS_DIR / f"{DATASET_NAME}_degree_distribution.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {output_path}")

    if show:
        plt.show()

    plt.close()



def plot_subgraph( G: nx.Graph, top_n: int = 300, 
                seed: int = 42, title: str = "Graph visualization", 
                show: bool = True,save: bool = False):
    """
    Prende i primi n nodi con il grado più alto e stampa il sottografo.
    Vengono isolati quindi i nodi più popolari e influenti della rete, ossia
    gli Hub.
    """
    
    if G.number_of_nodes() == 0:
        raise ValueError("Impossibile visualizzare un grafo vuoto.")

    if top_n <= 0:
        raise ValueError("top_n deve essere maggiore di zero.")


    # seleziona i nodi con grado più alto
    top_nodes = sorted(
        G.degree(),
        key=lambda item: item[1],
        reverse=True,
    )[:top_n]
    
    selected_nodes = [node for node, _ in top_nodes]

    # crea il sottografo con i nodi selezionati
    subgraph = G.subgraph(selected_nodes)

    """
    L'algoritmo spring_layout immagina che i nodi siano delle palline
    e gli archi delle molle. Esso, tira e spinge le palline finchè la
    rete non assume una forma chiara e bilanciata, riducendo gli 
    accavallamenti.
    """
    pos = nx.spring_layout(subgraph, seed=seed)

    fig, ax = plt.subplots(figsize=(12, 10))

    nx.draw_networkx_edges(subgraph, pos, alpha=0.15, width=0.5, ax=ax)
    nx.draw_networkx_nodes(subgraph, pos, node_size=15, ax=ax)
    
    ax.set_title(title)
    ax.axis("off")

    if save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        output_path = RESULTS_DIR / f"{DATASET_NAME}_subgraph.png"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {output_path}")

    if show:
        plt.show()

    plt.close(fig)

    return subgraph

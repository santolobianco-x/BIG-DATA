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
        print(f"Grafico salvato in: {output_path}")

    if show:
        plt.show()

    plt.close()


"""
MODULO DEDICATO ALLE BASELINE TRADIZIONALI

--> metodi che calcolano la probabilità di un futuro link basandosi
esclusivamente su regole statistiche locali.
Si dimostra l'utilità della GNN attraverso questi metodi.
"""
import networkx as nx
from math import log

def common_neighbors_scores(G: nx.Graph, edges):
    """
    Si analizzano i vicini in comune che hanno i due nodi, più vicini comuni hanno e più 
    più è probabile che si connetteranno in futuro.
    """
    return [len(set(G[u]) & set(G[v])) for u, v in edges]


def jaccard_scores(G: nx.Graph, edges):
    """
    Common Neighbors non è normalizzato e Jaccard risolve il problema dividendo per il totale
    dei vicini di entrambi i nodi.
    """
    return[
        score for _, _ , score in nx.jaccard_coefficient(G,edges)
    ]


def adamic_adar_scores(G: nx.Graph, edges):
    """
    Adami-Adar invece di contare semplicemente quanti amici in comune hanno due nodi,
    pesa ogni amico in comune in base a quanto è esclusivo.
    Minor collegamenti il nodo in comune ha, maggiore è la probabilità.
    """
    scores = []

    for u, v in edges:
        common = set(G.neighbors(u)) & set(G.neighbors(v))

        score = sum(
            1 / log(G.degree(w))
            for w in common
            if G.degree(w) > 1
        )

        scores.append(score)

    return scores # lista di scores pesati


def preferential_attachment_scores(G: nx.Graph, edges):
    """
    Non guarda i vicini in comune, bensì considera quelli che hanno grado alto e
    premia le coppie di nodi iper-connessi.
    """
    return[
        score for _, _, score in nx.preferential_attachment(G, edges)
    ]
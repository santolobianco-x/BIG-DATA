import networkx as nx
from math import log

def common_neighbors_scores(G: nx.Graph, edges):
    return [len(set(G[u]) & set(G[v])) for u, v in edges]


def jaccard_scores(G: nx.Graph, edges):
    return[
        score for _, _ , score in nx.jaccard_coefficient(G,edges)
    ]


def adamic_adar_scores(G: nx.Graph, edges):
    scores = []

    for u, v in edges:
        common = set(G.neighbors(u)) & set(G.neighbors(v))

        score = sum(
            1 / log(G.degree(w))
            for w in common
            if G.degree(w) > 1
        )

        scores.append(score)

    return scores


def preferential_attachment_scores(G: nx.Graph, edges):
    return[
        score for _, _, score in nx.preferential_attachment(G, edges)
    ]
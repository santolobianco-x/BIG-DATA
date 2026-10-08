import networkx as nx
import math

def common_neighbors_scores(G: nx.Graph, edges):
    scores = []


    for u, v in edges:
        score = len(list(nx.common_neighbors(G, u, v)))
        scores.append(score)

    return scores


def jaccard_scores(G: nx.Graph, edges):
    return[
        score for _, _ , score in nx.jaccard_coefficient(G,edges)
    ]



def adamic_adar_scores(G: nx.Graph, edges):

    scores = []

    for u, v in edges:

        score = 0.0

        for w in nx.common_neighbors(G, u, v):

            degree = G.degree(w)

            if degree > 1:
                score += 1 / math.log(degree)

        scores.append(score)

    return scores


def preferential_attachment_scores(G: nx.Graph, edges):
    return[
        score for _, _, score in nx.preferential_attachment(G, edges)
    ]
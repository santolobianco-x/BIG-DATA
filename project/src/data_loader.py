"""
MODULO RESPONSABILE DEL CARICAMENTO INIZIALE DEI DATI
"""
import networkx as nx


def load_graph(path):
    """
    Carica il dataset come grafo non orientato

    Il file sorgente deve essere una 'edge list', ovvero un file testuale
    dove ogni riga rappreetna un arco tra due nodi.

    """

    # nx.read_edgelist legge automaticamente file compressi
    G = nx.read_edgelist(
        path,
        nodetype=int, # assicura che gli ID dei nodi siano trattati come interi e non come stringhe
        comments='#' # ignora le righe di intestazione del dataset
    )

    
    G.remove_edges_from(nx.selfloop_edges(G)) # rimozione self-loop(nelle reti sociali un nodo non può formare un link con sé stesso)

    return G #oggetto NetworkX che rappresenta il grafo non orientato.
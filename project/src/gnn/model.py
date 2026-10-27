"""
MODULO CONTENTENTE L'ARCHITETTURA DELLA GRAPH NEURAL NETWORK

--> implementa GraphSAGE(sample and aggregate) che a differenza della
    GNN classica, GraphSAGE è progettata per essere scalabile e per 
    generalizzare meglio imparando funzioni di aggregazione.
"""
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class GraphSAGE(torch.nn.Module):
    """
    Obiettivo del modello è prendere in input le feature iniziali dei nodi
    e restituire in output dei vettori densi contenenti informazioni sul nodo
    e la struttura del vicinato.
    """
    def __init__(self, in_channels, hidden_channels,out_channels):
        super().__init__()# INIZIALIZZA LA CLASSE MADRE
        
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        # Aggrega i vicini a distanza 1
        self.conv2 = SAGEConv(hidden_channels,out_channels)
        # Aggrega i vicini a distanza 2
        """
        Con due layer, ogni nodo raccoglie informazioni dai suoi
        amici stretti(1^layer) e dagli amici dei suoi amici(2^layer)
        """

    def forward(self, x, edge_index):
        """
        Definisce il passaggio in avanti dei dati alla rete.
        - x: matrice delle feature dei nodi [numero_nodi, in_channels]
        - edge_index: matrice delle connessioni.
        """


        # ogni nodo guarda i propri vicini diretti e aggiorna il proprio stato
        x = self.conv1(x, edge_index)
        # funzione di attivazione che permette di apprendere relazioni complesse
        x = F.relu(x)
        # ogni nodo guarda i vicini dei vicini e aggiorna lo stato
        x = self.conv2(x, edge_index)
        return x #embedding finale di ciascun nodo
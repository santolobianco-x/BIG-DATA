"""
MODULO CONTENENTE I DECODER PER LA LINK PREDICTION

--> nell'architettura Encoder-Decoder per grafi:
        - encoder --> genera gli embedding dei nodi nello spazio latente
        - decoder --> prende coppie di embedding e restituisce uno score
                        che indica la probabilità di un collegamento tra nodi.
"""
import torch
import torch.nn as nn


class MLPDecoder(nn.Module):
    """
    MLPDecoder è un decoder basato su un Multi-Layer Perceptron, ossia decoder
    che ha pesi addestrabili che permettono al modello di imparare una funzione
    di similarità complessa e non-lineare tra i nodi.
    """

    def __init__(self, embedding_dim, hidden_dim):
        # embedding_dim --> dimensione vettori di embedding in uscita dall'encoder
        # hidden_dim --> neuroni nello stato nascosto del decoder
        super().__init__()

        self.mlp = nn.Sequential(
            # 1^ layer che prende in input feature di dimensione doppia
            # si combinano i due vettori di dimensione embedding_dim
            nn.Linear(embedding_dim * 2, hidden_dim),
            # funzione di attivazione
            nn.ReLU(),
            # 2^ layer che comprime il vettore nascosto in un singolo numero(score)
            nn.Linear(hidden_dim, 1)
        )


    def forward(self, z, edge_index):
        # Passaggio in avanti per calcolare lo score di un set di archi
        
        edge_index = edge_index.long()

        # separazione nodi sorgente e nodi destinazione
        src = edge_index[0]
        dst = edge_index[1]

        # estrae gli embedding corrispondenti a sorgenti e destinazioni
        z_src = z[src]
        z_dst = z[dst]
        """
        Poichè il grafo non è orienatto, si utilizzano operatori commutativi/simmetrici
        per combinare gli embedding. L'arco (A, B) avrà lo stesso input dell'arco
        (B, A).
        """
        product = z_src * z_dst # prodotto che cattura la similarità

        difference = torch.abs(z_src - z_dst) # calcola la diversità strutturale attraverso la differenza assoluta

        # si concatenano le due feature per creare il vettore che descrive l'arco
        edge_features = torch.cat([product, difference],dim=1)
        
        return self.mlp(edge_features).view(-1) # si ottiene un vettore 1D di logits attraverso 
        # .view(-1) che appiattisce l'output



class DotProductDecoder(nn.Module):
    """
    Decoder basato sul prodotto scalare tra le embedding dei due nodi.
    Se due nodi sono simili, i vettori di embedding puntano nella stessa direzione
    nello spazio latente e il prodotto scalare restituirà un numero alto.
    Non ha parametri da addestrare e può risultare più veloce.
    """

    def forward(self, z, edge_index):
        edge_index = edge_index.long()

        src = edge_index[0]
        dst = edge_index[1]

        z_src = z[src]
        z_dst = z[dst]

        # È matematicamente equivalente alla similarità del coseno non normalizzata.
        return torch.sum(z_src * z_dst, dim=1) # dato che è un prodotto scalare si fa la somma finale.

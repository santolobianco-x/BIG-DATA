from collections import defaultdict



class GridNetwork:

    def __init__(self, n):
        self.n = n
        self.long_links = defaultdict(list)

    # RETE SU UNA GRIGLIA n x n
    # Ogni nodo è rappresentato dalla coppia (i, j)


    def neighbors_local(self, node):

        i, j = node
        neighbors = []

        # Restituisce i vicini locali del nodo

        for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            r = i + di
            c = j + dj

            if 0 <= r < self.n and 0 <= c < self.n:
                neighbors.append((r, c))

        return neighbors


    @staticmethod
    def manhattan(a, b):
        # Calcola la distanza di Manhattan tra due nodi
        return abs(a[0] - b[0])+ abs(a[1] - b[1])


    def all_neighbors(self, node):
        # Ritorna i vicini locali + vicini a lunga distanza
        return self.neighbors_local(node) + self.long_links.get(node, [])


    def add_long_link(self, src, dst):
        # Aggiunge collegamenti a lunga distanza
        self.long_links[src].append(dst)
        self.long_links[dst].append(src)
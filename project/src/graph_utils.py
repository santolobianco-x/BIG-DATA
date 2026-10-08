import networkx as nx
import matplotlib.pyplot as plt



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



def plot_degree_distribution(G: nx.Graph, title):
    degrees = [degree for _, degree in G.degree()]

    plt.figure(figsize=(8,5))
    plt.hist(degrees, bins=50)
    plt.xlabel("Degree")
    plt.ylabel("Number of nodes")
    plt.title(title)
    plt.show()


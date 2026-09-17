import numpy as np

from graph_utils import (
    load_graph,
    print_graph_info,
    get_in_degree,
    get_out_degree,
    plot_and_fit,
    get_largest_component,
    get_pagerank,
    cosine_similarity
)

from graph_models import (
    generate_random_graph,
    find_ba_m,
    generate_ba_graph
)


G = load_graph()
print_graph_info(G, True)


in_degrees = get_in_degree(G)
out_degrees = get_out_degree(G)

plot_and_fit(in_degrees, "Distribuzione In-Degree (log-log)")
plot_and_fit(out_degrees, "Distribuzione Out-Degree (log-log)")


Gcc = get_largest_component(G)


print()
print("Componente debolmente connessa più grande:")
print_graph_info(Gcc, True)



pagerank_scores = get_pagerank(Gcc)

sorted_pagerank = sorted(
    pagerank_scores.items(),
    key=lambda item: item[1],
    reverse=True
)

print("\nTop 10 nodi per PageRank:")

for node, score in sorted_pagerank[:10]:
    print(f"Nodo {node}: PageRank = {score:.6f}")



n = len(Gcc.nodes)
E_original = len(Gcc.edges)


G_random = generate_random_graph(n)


m = find_ba_m(n, E_original)
G_BA = generate_ba_graph(n, m)



print("\nRandom Graph:")
print_graph_info(G_random, False)


print("\nBarabasi-Albert:")
print_graph_info(G_BA, False)


print("\nArchi componente originale:", E_original)
print("m scelto:", m)
print("Archi Barabasi-Albert:", len(G_BA.edges))



pagerank_random = get_pagerank(G_random)
pagerank_BA = get_pagerank(G_BA)



original_vector = np.array(
    sorted(pagerank_scores.values(), reverse=True)
)

random_vector = np.array(
    sorted(pagerank_random.values(), reverse=True)
)

BA_vector = np.array(
    sorted(pagerank_BA.values(), reverse=True)
)



cosine_random = cosine_similarity(original_vector, random_vector)

cosine_BA = cosine_similarity(original_vector, BA_vector)


print()
print("Cosine similarity:")
print("Random Graph:", cosine_random)
print("Barabasi-Albert:", cosine_BA)
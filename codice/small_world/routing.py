from grid_network import GridNetwork


def greedy_route(net: GridNetwork, source, target, max_steps=10000):
    """
    Esegue il routing greedy.
    Ad ogni passo viene scelto il vicino più vicino alla destinazione
    """


    current = source

    path = [current]

    optimal = GridNetwork.manhattan(
        source,
        target
    )

    while current != target and len(path) <= max_steps:

        neighbors = net.all_neighbors(current)

        if not neighbors:
            break

        best = min(neighbors, key= lambda v : GridNetwork.manhattan(v, target))

        current_distance = GridNetwork.manhattan(current, target)

        best_distance = GridNetwork.manhattan(best, target)


        if best_distance >= current_distance:
            break

        current = best
        path.append(current)


    return {
        "path": path,
        "steps": len(path)-1,
        "reached": current == target,
        "optimal": optimal
    }
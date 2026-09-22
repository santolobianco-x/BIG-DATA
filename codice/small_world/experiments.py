import random
import time
import numpy as np


from graph_models import build_watts_strogatz, build_kleinberg
from routing import greedy_route


def measure_routing_performance(n, model="kleinberg", alpha=2.0, num_trials=50, num_links_per_node=1, seed=42):
    # Misura le prestazioni del greedy routing

    random.seed(seed)
    np.random.seed(seed)

    if model == "watts_strogatz":
        net = build_watts_strogatz(n, num_links_per_node, seed)

    elif model == "kleinberg":
        net = build_kleinberg(n, num_links_per_node, alpha, seed)

    else:
        raise ValueError("Modello Sconosciuto")

    steps_all = []
    optimal_all = []

    reached = 0

    for _ in range(num_trials):
        src = (
            random.randint(0, n-1),
            random.randint(0, n-1)
        )

        while True:
            dst = (
                random.randint(0, n-1),
                random.randint(0, n-1)
            )

            if dst != src:
                break

        result = greedy_route(net, src, dst)


        steps_all.append(result["steps"])
        optimal_all.append(result["optimal"])

        if result["reached"]:
            reached += 1

    return{
        "steps_mean": float(np.mean(steps_all)),
        "steps_median": float(np.median(steps_all)),
        "steps_all": steps_all,
        "reach_rate": reached/ num_trials,
        "optimal_mean": float(np.mean(optimal_all))
    }

def scaling_experiment(n_values, num_trials = 30, seed = 42):
    # Esegue l'esperimento di scaling per entrambi i modelli

    results = {
        "ws": [],
        "kl": []
    }

    for n in n_values:
        print(f"n={n}x{n}={n*n} nodi...",
              end=" ",
              flush=True)
        start = time.time()


        results["ws"].append(
            measure_routing_performance(n, "watts_strogatz", num_trials=num_trials, seed=seed)
        )


        results["kl"].append(
            measure_routing_performance(n, "kleinberg", alpha=2.0, num_trials=num_trials, seed=seed)
        )

        print(f"Fatto in {time.time()}")

    return results
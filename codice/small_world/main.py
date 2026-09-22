import math

from graph_models import (
    build_watts_strogatz,
    build_kleinberg
)

from routing import greedy_route

from experiments import scaling_experiment

from plots import (
    plot_scaling,
    plot_alpha_sweep
)


if __name__ == "__main__":

    print("=" * 60)
    print("Navigabilità nelle reti Small World")
    print("=" * 60)

  

    print("\n[1] Test rapido")

    net_ws = build_watts_strogatz(
        10,
        seed=0
    )

    net_kl = build_kleinberg(
        10,
        alpha=2,
        seed=0
    )

    source = (0, 0)
    target = (9, 9)

    result_ws = greedy_route(
        net_ws,
        source,
        target
    )

    result_kl = greedy_route(
        net_kl,
        source,
        target
    )

    print(
        "Watts-Strogatz:",
        result_ws["steps"],
        "passi"
    )

    print(
        "Kleinberg:",
        result_kl["steps"],
        "passi"
    )

   

    print("\n[2] Esperimento di scaling")

    n_values = [
        5, 8, 10, 12,
        15, 18, 20,
        25, 30
    ]

    results = scaling_experiment(
        n_values,
        num_trials=40,
        seed=42
    )

    plot_scaling(
        n_values,
        results
    )

    
    print("\nRisultati:")

    print(
        f"{'n':>5} "
        f"{'WS':>10} "
        f"{'Kleinberg':>10}"
    )

    for i, n in enumerate(n_values):

        ws = results["ws"][i]["steps_mean"]
        kl = results["kl"][i]["steps_mean"]

        print(
            f"{n:>5} "
            f"{ws:>10.1f} "
            f"{kl:>10.1f}"
        )

    # ==========================================================
    # BONUS
    # ==========================================================

    print("\n[3] BONUS: variazione di alpha")

    plot_alpha_sweep(
        n=20,
        num_trials=40,
        seed=42
    )

    print("\nEsercizio completato.")
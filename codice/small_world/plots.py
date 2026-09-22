import numpy as np
import matplotlib.pyplot as plt

from experiments import measure_routing_performance


def plot_scaling(n_values, results):

    n = np.array(n_values)

    ws = np.array([
        r["steps_mean"]
        for r in results["ws"]
    ])

    kl = np.array([
        r["steps_mean"]
        for r in results["kl"]
    ])

    ws_theory = n ** (2 / 3)
    kl_theory = np.log2(n) ** 2

    
    ws_theory *= ws.max() / ws_theory.max()
    kl_theory *= kl.max() / kl_theory.max()

    plt.figure(figsize=(10, 5))

    plt.loglog(
        n,
        ws,
        "o-",
        label="Watts-Strogatz"
    )

    plt.loglog(
        n,
        kl,
        "s-",
        label="Kleinberg"
    )

    plt.loglog(
        n,
        ws_theory,
        "--",
        label="n^(2/3)"
    )

    plt.loglog(
        n,
        kl_theory,
        "--",
        label="log²(n)"
    )

    plt.xlabel("n")
    plt.ylabel("Passi medi")

    plt.title(
        "Scaling del routing greedy"
    )

    plt.legend()
    plt.grid(True, which="both")

    plt.tight_layout()

    plt.savefig(
        "small_world_scaling.png",
        dpi=150
    )

    plt.show()


def plot_alpha_sweep(
    n=20,
    alphas=None,
    num_trials=40,
    seed=42
):

    if alphas is None:
        alphas = np.linspace(
            0,
            4,
            17
        )

    steps = []

    for alpha in alphas:

        result = measure_routing_performance(
            n,
            model="kleinberg",
            alpha=alpha,
            num_trials=num_trials,
            seed=seed
        )

        steps.append(
            result["steps_mean"]
        )

    plt.figure(figsize=(7, 4))

    plt.plot(
        alphas,
        steps,
        "o-"
    )

    plt.axvline(
        2,
        linestyle="--",
        label="alpha = 2"
    )

    plt.xlabel("alpha")
    plt.ylabel("Passi medi")

    plt.title(
        f"Variazione di alpha - griglia {n}x{n}"
    )

    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        "alpha_sweep.png",
        dpi=150
    )

    plt.show()
"""
Obiettivo
---------
Implementare da zero due modelli di rete su griglia n×n
(Watts-Strogatz e Kleinberg), simulare il routing greedy su
entrambi e verificare empiricamente i bound teorici:

    Watts-Strogatz  →  O(n^(2/3)) passi attesi
    Kleinberg       →  O(log²n)   passi attesi

Struttura dell'esercizio
------------------------
Parte 1 — Rappresentazione della rete su griglia
Parte 2 — Generazione dei l\ink a lunga distanza
Parte 3 — Routing greedy decentralizzato
Parte 4 — Misura sperimentale delle prestazioni
Parte 5 — Visualizzazione e confronto con i bound teorici
Bonus   — Variazione dell'esponente α e ricerca del punto ottimale

Prerequisiti:
    numpy
    matplotlib

Installazione:
    pip install numpy matplotlib
"""

import math
import random
import time
from collections import defaultdict
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ── Costanti di colore per i plot ────────────────────────────────────────────

C_SW = "#E24B4A"      # Watts-Strogatz
C_KL = "#1D9E75"      # Kleinberg
C_OPT = "#378ADD"     # ottimale


# ==============================================================================
# PARTE 1 — Rappresentazione della rete su griglia
# ==============================================================================

class GridNetwork:
    """
    Rete su griglia n×n con:

    - link locali: i 4 vicini di Manhattan
    - link a lunga distanza

    I nodi sono identificati dalla coppia (i, j), con:

        0 ≤ i, j < n

    I link a lunga distanza sono memorizzati nel dizionario:

        long_links[(i, j)] = [(r1, c1), (r2, c2), ...]

    I link lunghi sono non orientati.
    """

    def __init__(self, n: int):
        self.n = n

        # Dizionario:
        # nodo → lista delle destinazioni dei link lunghi
        self.long_links: dict[tuple, list[tuple]] = defaultdict(list)

    # ------------------------------------------------------------------
    # TODO 1.1
    # ------------------------------------------------------------------

    def neighbors_local(self, node: tuple) -> list[tuple]:
        """
        Restituisce i vicini locali di Manhattan del nodo (i, j).

        I possibili vicini sono:

            (i+1, j)
            (i-1, j)
            (i, j+1)
            (i, j-1)

        Vengono esclusi i nodi fuori dalla griglia.

        Parametri
        ---------
        node : tuple
            Nodo della forma (i, j).

        Ritorna
        -------
        list[tuple]
            Lista dei vicini locali.
        """

        i, j = node

        neighbors = []

        # Spostamenti:
        # giù, su, destra, sinistra
        directions = [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1)
        ]

        for di, dj in directions:

            r = i + di
            c = j + dj

            # Manteniamo solo i nodi dentro la griglia
            if 0 <= r < self.n and 0 <= c < self.n:
                neighbors.append((r, c))

        return neighbors

    # ------------------------------------------------------------------
    # TODO 1.2
    # ------------------------------------------------------------------

    @staticmethod
    def manhattan(a: tuple, b: tuple) -> int:
        """
        Calcola la distanza di Manhattan tra due nodi.

        La distanza di Manhattan è:

            d(a,b) = |a_i - b_i| + |a_j - b_j|

        Parametri
        ---------
        a, b : tuple
            Nodi della forma (i, j).

        Ritorna
        -------
        int
            Distanza di Manhattan.
        """

        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    # ------------------------------------------------------------------

    def all_neighbors(self, node: tuple) -> list[tuple]:
        """
        Restituisce tutti i vicini del nodo:

        - vicini locali
        - vicini raggiungibili tramite link lunghi
        """

        return self.neighbors_local(node) + self.long_links.get(node, [])

    # ------------------------------------------------------------------

    def add_long_link(self, src: tuple, dst: tuple):
        """
        Aggiunge un link a lunga distanza non orientato.

        Se aggiungiamo:

            src → dst

        aggiungiamo anche:

            dst → src
        """

        self.long_links[src].append(dst)
        self.long_links[dst].append(src)


# ==============================================================================
# PARTE 2 — Generazione dei link a lunga distanza
# ==============================================================================

def build_watts_strogatz(
        n: int,
        num_links_per_node: int = 1,
        seed: Optional[int] = None
) -> GridNetwork:
    """
    Costruisce una GridNetwork con link a lunga distanza
    scelti uniformemente a caso.

    Per ogni nodo vengono scelti num_links_per_node
    nodi destinazione casuali.

    La distribuzione è quindi uniforme:

        P(u → v) = 1 / (n² - 1)

    per ogni v diverso da u.

    Parametri
    ---------
    n : int
        Lato della griglia.

    num_links_per_node : int
        Numero di link lunghi per nodo.

    seed : int, opzionale
        Seme per la generazione casuale.

    Ritorna
    -------
    GridNetwork
        La rete costruita.
    """

    if seed is not None:
        random.seed(seed)

    net = GridNetwork(n)

    # Iteriamo su tutti i nodi
    for i in range(n):
        for j in range(n):

            src = (i, j)

            # Creiamo i link lunghi del nodo
            for _ in range(num_links_per_node):

                while True:

                    # Destinazione casuale
                    dst = (
                        random.randint(0, n - 1),
                        random.randint(0, n - 1)
                    )

                    # Non possiamo creare un link verso se stessi
                    if dst != src:
                        break

                # Aggiungiamo il link
                net.add_long_link(src, dst)

    return net


# ------------------------------------------------------------------------------

def build_kleinberg(
        n: int,
        num_links_per_node: int = 1,
        alpha: float = 2.0,
        seed: Optional[int] = None
) -> GridNetwork:
    """
    Costruisce una GridNetwork secondo il modello di Kleinberg.

    La probabilità di scegliere un nodo v partendo da u è
    proporzionale a:

        d(u,v)^(-alpha)

    cioè:

        P(u → v) =
            d(u,v)^(-alpha)
            -------------------
            somma_w d(u,w)^(-alpha)

    Per una griglia bidimensionale il valore teoricamente
    ottimale è:

        alpha = 2

    Quando alpha = 0 tutti i pesi sono uguali e quindi
    la distribuzione diventa uniforme.

    Parametri
    ---------
    n : int
        Lato della griglia.

    num_links_per_node : int
        Numero di link lunghi per nodo.

    alpha : float
        Esponente della distribuzione.

    seed : int, opzionale
        Seme per la generazione casuale.

    Ritorna
    -------
    GridNetwork
        La rete costruita.
    """

    if seed is not None:
        np.random.seed(seed)

    net = GridNetwork(n)

    # Iteriamo su tutti i nodi sorgente
    for i in range(n):
        for j in range(n):

            src = (i, j)

            candidates = []
            weights = []

            # Costruiamo l'insieme dei possibili destinatari
            for r in range(n):
                for c in range(n):

                    dst = (r, c)

                    # Un nodo non può creare un link verso se stesso
                    if dst == src:
                        continue

                    # Distanza di Manhattan
                    d = GridNetwork.manhattan(src, dst)

                    # Peso d^(-alpha)
                    weight = d ** (-alpha)

                    candidates.append(dst)
                    weights.append(weight)

            # Convertiamo i pesi in array numpy
            weights = np.array(weights, dtype=float)

            # Normalizzazione:
            #
            # somma dei pesi = 1
            #
            weights /= weights.sum()

            # Creiamo i link lunghi
            for _ in range(num_links_per_node):

                # Estraiamo un indice secondo la distribuzione
                index = np.random.choice(
                    len(candidates),
                    p=weights
                )

                dst = candidates[index]

                # Aggiungiamo il link
                net.add_long_link(src, dst)

    return net


# ==============================================================================
# PARTE 3 — Routing greedy decentralizzato
# ==============================================================================

def greedy_route(
        net: GridNetwork,
        source: tuple,
        target: tuple,
        max_steps: int = 10_000
) -> dict:
    """
    Esegue il routing greedy dal nodo source al nodo target.

    Ad ogni passo il nodo corrente:

    1. guarda solamente i propri vicini;
    2. calcola la distanza Manhattan di ogni vicino dal target;
    3. sceglie il vicino più vicino al target.

    Non viene utilizzata nessuna informazione globale.

    Parametri
    ---------
    net : GridNetwork
        La rete.

    source : tuple
        Nodo di partenza.

    target : tuple
        Nodo destinazione.

    max_steps : int
        Numero massimo di passi.

    Ritorna
    -------
    dict
        Dizionario contenente:

        path
            Lista dei nodi visitati.

        steps
            Numero di passi effettuati.

        reached
            True se il target è stato raggiunto.

        optimal
            Distanza Manhattan tra source e target.
    """

    # Nodo corrente
    cur = source

    # Percorso
    path = [cur]

    # Distanza minima teoricamente possibile:
    # distanza Manhattan tra source e target
    optimal = GridNetwork.manhattan(source, target)

    # Continuiamo finché non arriviamo al target
    while cur != target and len(path) <= max_steps:

        # Otteniamo tutti i vicini:
        # locali + link lunghi
        neighbors = net.all_neighbors(cur)

        # Se non ci sono vicini ci fermiamo
        if not neighbors:
            break

        # Troviamo il vicino che minimizza la distanza
        # Manhattan dal target
        best = min(
            neighbors,
            key=lambda v: GridNetwork.manhattan(v, target)
        )

        # Distanza del nodo corrente dal target
        current_distance = GridNetwork.manhattan(
            cur,
            target
        )

        # Distanza del miglior vicino dal target
        best_distance = GridNetwork.manhattan(
            best,
            target
        )

        # Se il miglior vicino non è più vicino
        # siamo bloccati in un minimo locale.
        if best_distance >= current_distance:
            break

        # Ci spostiamo al nodo migliore
        cur = best

        # Aggiungiamo il nodo al percorso
        path.append(cur)

    # Numero di archi attraversati
    steps = len(path) - 1

    return {
        "path": path,
        "steps": steps,
        "reached": cur == target,
        "optimal": optimal
    }


# ==============================================================================
# PARTE 4 — Misura sperimentale delle prestazioni
# ==============================================================================

def measure_routing_performance(
        n: int,
        model: str = "kleinberg",
        alpha: float = 2.0,
        num_trials: int = 50,
        num_links_per_node: int = 1,
        seed: Optional[int] = 42
) -> dict:
    """
    Genera una rete del modello scelto e misura le prestazioni
    del routing greedy su num_trials coppie source-target.

    Ritorna:

        steps_mean
            Media dei passi.

        steps_median
            Mediana dei passi.

        steps_all
            Lista dei passi di tutti i trial.

        reach_rate
            Percentuale di trial in cui il target viene raggiunto.

        optimal_mean
            Distanza Manhattan media.
    """

    # Impostiamo i semi
    random.seed(seed)
    np.random.seed(seed)

    # ------------------------------------------------------------------
    # Costruzione della rete
    # ------------------------------------------------------------------

    if model == "watts_strogatz":

        net = build_watts_strogatz(
            n,
            num_links_per_node,
            seed=seed
        )

    elif model == "kleinberg":

        net = build_kleinberg(
            n,
            num_links_per_node,
            alpha=alpha,
            seed=seed
        )

    else:

        raise ValueError(
            f"Modello sconosciuto: {model}"
        )

    # Liste per memorizzare i risultati
    steps_all = []
    optimal_all = []

    # Numero di routing riusciti
    reached = 0

    # ------------------------------------------------------------------
    # Esecuzione dei trial
    # ------------------------------------------------------------------

    for _ in range(num_trials):

        # Sorgente casuale
        src = (
            random.randint(0, n - 1),
            random.randint(0, n - 1)
        )

        # Destinazione casuale diversa dalla sorgente
        while True:

            dst = (
                random.randint(0, n - 1),
                random.randint(0, n - 1)
            )

            if dst != src:
                break

        # Eseguiamo il greedy routing
        result = greedy_route(
            net,
            src,
            dst
        )

        # Salviamo i risultati
        steps_all.append(result["steps"])
        optimal_all.append(result["optimal"])

        if result["reached"]:
            reached += 1

    # ------------------------------------------------------------------
    # Risultati finali
    # ------------------------------------------------------------------

    return {
        "steps_mean": float(np.mean(steps_all)),
        "steps_median": float(np.median(steps_all)),
        "steps_all": steps_all,
        "reach_rate": reached / num_trials,
        "optimal_mean": float(np.mean(optimal_all))
    }


# ------------------------------------------------------------------------------

def scaling_experiment(
        n_values: list[int],
        num_trials: int = 30,
        seed: int = 42
) -> dict:
    """
    Esegue measure_routing_performance per ogni valore di n.

    Vengono confrontati:

        Watts-Strogatz
        Kleinberg con alpha = 2

    Ritorna:

        {
            "ws": [...],
            "kl": [...]
        }
    """

    results = {
        "ws": [],
        "kl": []
    }

    # Per ogni dimensione della griglia
    for n in n_values:

        print(
            f"  n={n}x{n}={n*n} nodi...",
            end=" ",
            flush=True
        )

        t0 = time.time()

        # Watts-Strogatz
        results["ws"].append(
            measure_routing_performance(
                n,
                "watts_strogatz",
                num_trials=num_trials,
                seed=seed
            )
        )

        # Kleinberg
        results["kl"].append(
            measure_routing_performance(
                n,
                "kleinberg",
                alpha=2.0,
                num_trials=num_trials,
                seed=seed
            )
        )

        elapsed = time.time() - t0

        print(
            f"fatto in {elapsed:.1f}s"
        )

    return results


# ==============================================================================
# PARTE 5 — Visualizzazione
# ==============================================================================

def plot_scaling(
        n_values: list[int],
        results: dict
):
    """
    Produce due grafici.

    Grafico A
    ---------
    Passi medi del routing greedy rispetto a n
    in scala log-log.

    Vengono sovrapposte le curve teoriche:

        n^(2/3)

    e:

        log²(n)

    Grafico B
    ---------
    Rapporto:

        passi_greedy / distanza_ottimale

    """

    n_arr = np.array(n_values)

    # Risultati Watts-Strogatz
    sw_steps = np.array([
        r["steps_mean"]
        for r in results["ws"]
    ])

    # Risultati Kleinberg
    kl_steps = np.array([
        r["steps_mean"]
        for r in results["kl"]
    ])

    # Distanze ottimali
    sw_opt = np.array([
        r["optimal_mean"]
        for r in results["ws"]
    ])

    kl_opt = np.array([
        r["optimal_mean"]
        for r in results["kl"]
    ])

    # ------------------------------------------------------------------
    # Curve teoriche
    # ------------------------------------------------------------------

    theory_sw = n_arr ** (2 / 3)

    theory_kl = np.log2(n_arr) ** 2

    # Riscalamento delle curve teoriche per poterle
    # confrontare visivamente con i dati
    if sw_steps.max() > 0:

        theory_sw = (
            theory_sw *
            sw_steps.max() /
            theory_sw.max()
        )

    if kl_steps.max() > 0:

        theory_kl = (
            theory_kl *
            kl_steps.max() /
            theory_kl.max()
        )

    # ------------------------------------------------------------------
    # Creazione figura
    # ------------------------------------------------------------------

    fig = plt.figure(
        figsize=(12, 5)
    )

    gs = gridspec.GridSpec(
        1,
        2,
        wspace=0.35
    )

    # ==================================================================
    # GRAFICO A
    # ==================================================================

    ax1 = fig.add_subplot(gs[0])

    ax1.loglog(
        n_arr,
        sw_steps,
        "o-",
        color=C_SW,
        lw=2,
        label="Watts-Strogatz (α=0)"
    )

    ax1.loglog(
        n_arr,
        kl_steps,
        "s-",
        color=C_KL,
        lw=2,
        label="Kleinberg (α=2)"
    )

    ax1.loglog(
        n_arr,
        theory_sw,
        "--",
        color=C_SW,
        alpha=0.5,
        lw=1.5,
        label="n^(2/3) teorico"
    )

    ax1.loglog(
        n_arr,
        theory_kl,
        "--",
        color=C_KL,
        alpha=0.5,
        lw=1.5,
        label="log²(n) teorico"
    )

    ax1.set_xlabel(
        "n  (lato della griglia)",
        fontsize=11
    )

    ax1.set_ylabel(
        "Passi medi del routing greedy",
        fontsize=11
    )

    ax1.set_title(
        "Scaling del routing greedy (log-log)",
        fontsize=12
    )

    ax1.legend(
        fontsize=9
    )

    ax1.grid(
        True,
        which="both",
        alpha=0.3
    )

    # ==================================================================
    # GRAFICO B
    # ==================================================================

    ax2 = fig.add_subplot(gs[1])

    ax2.plot(
        n_arr,
        sw_steps / sw_opt,
        "o-",
        color=C_SW,
        lw=2,
        label="Watts-Strogatz"
    )

    ax2.plot(
        n_arr,
        kl_steps / kl_opt,
        "s-",
        color=C_KL,
        lw=2,
        label="Kleinberg"
    )

    # Linea dell'ottimo
    ax2.axhline(
        1,
        color="gray",
        lw=1,
        linestyle=":",
        label="ottimale (=1)"
    )

    ax2.set_xlabel(
        "n  (lato della griglia)",
        fontsize=11
    )

    ax2.set_ylabel(
        "Passi greedy / distanza Manhattan",
        fontsize=11
    )

    ax2.set_title(
        "Efficienza del routing greedy",
        fontsize=12
    )

    ax2.legend(
        fontsize=9
    )

    ax2.grid(
        True,
        alpha=0.3
    )

    # ------------------------------------------------------------------
    # Titolo generale
    # ------------------------------------------------------------------

    plt.suptitle(
        "Navigabilità: Watts-Strogatz vs Kleinberg",
        fontsize=13,
        fontweight="bold",
        y=1.02
    )

    plt.tight_layout()

    # Salvataggio
    plt.savefig(
        "small_world_scaling.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.show()

    print(
        "Grafico salvato in small_world_scaling.png"
    )


# ==============================================================================
# BONUS — Variazione di alpha
# ==============================================================================

def plot_alpha_sweep(
        n: int = 20,
        alphas: list = None,
        num_trials: int = 40,
        seed: int = 42
):
    """
    BONUS

    Misura i passi medi del routing greedy al variare
    dell'esponente alpha.

    L'obiettivo è osservare il comportamento intorno a:

        alpha = 2

    che è il valore teoricamente ottimale per una
    griglia bidimensionale.

    Parametri
    ---------
    n : int
        Lato della griglia.

    alphas : list
        Valori di alpha da testare.

    num_trials : int
        Numero di routing per ogni alpha.

    seed : int
        Seme casuale.
    """

    if alphas is None:

        alphas = np.linspace(
            0,
            4,
            17
        ).tolist()

    print(
        f"\nBONUS: sweep su α per n={n}..."
    )

    steps = []

    # Testiamo ogni alpha
    for a in alphas:

        result = measure_routing_performance(
            n,
            "kleinberg",
            alpha=a,
            num_trials=num_trials,
            seed=seed
        )

        steps.append(
            result["steps_mean"]
        )

        print(
            f"  α={a:.2f} → "
            f"{result['steps_mean']:.1f} passi medi"
        )

    # ------------------------------------------------------------------
    # Grafico
    # ------------------------------------------------------------------

    plt.figure(
        figsize=(7, 4)
    )

    plt.plot(
        alphas,
        steps,
        "o-",
        color="#534AB7",
        lw=2
    )

    # Linea verticale in corrispondenza di alpha = 2
    plt.axvline(
        2.0,
        color=C_KL,
        lw=1.5,
        linestyle="--",
        label="α=2"
    )

    plt.xlabel(
        "Esponente α",
        fontsize=11
    )

    plt.ylabel(
        "Passi medi del routing greedy",
        fontsize=11
    )

    plt.title(
        f"Variazione di α — griglia {n}×{n}",
        fontsize=12
    )

    plt.legend(
        fontsize=10
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    # Salvataggio
    plt.savefig(
        "alpha_sweep.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.show()

    print(
        "Grafico salvato in alpha_sweep.png"
    )


# ==============================================================================
# MAIN — esecuzione dell'esercizio completo
# ==============================================================================

if __name__ == "__main__":

    print("=" * 60)

    print(
        "ESERCIZIO: Navigabilità nelle reti Small World"
    )

    print("=" * 60)

    # ==================================================================
    # TEST RAPIDO
    # ==================================================================

    print(
        "\n[1] Test rapido su griglia 10×10"
    )

    try:

        # Costruiamo una rete Watts-Strogatz
        net_sw = build_watts_strogatz(
            10,
            seed=0
        )

        # Costruiamo una rete Kleinberg
        net_kl = build_kleinberg(
            10,
            alpha=2.0,
            seed=0
        )

        # Source e target
        src = (0, 0)
        dst = (9, 9)

        # Routing Watts-Strogatz
        r_sw = greedy_route(
            net_sw,
            src,
            dst
        )

        # Routing Kleinberg
        r_kl = greedy_route(
            net_kl,
            src,
            dst
        )

        print(
            f"  WS: {r_sw['steps']} passi "
            f"(ottimale: {r_sw['optimal']}) "
            f"raggiunto: {r_sw['reached']}"
        )

        print(
            f"  Kleinberg: {r_kl['steps']} passi "
            f"(ottimale: {r_kl['optimal']}) "
            f"raggiunto: {r_kl['reached']}"
        )

    except NotImplementedError as e:

        print(
            f"  [TODO] {e}"
        )

    # ==================================================================
    # ESPERIMENTO DI SCALING
    # ==================================================================

    print(
        "\n[2] Esperimento di scaling"
    )

    N_VALUES = [
        5,
        8,
        10,
        12,
        15,
        18,
        20,
        25,
        30
    ]

    NUM_TRIALS = 40

    try:

        print(
            f"  Valori di n: {N_VALUES}"
        )

        results = scaling_experiment(
            N_VALUES,
            num_trials=NUM_TRIALS,
            seed=42
        )

        # Creazione grafici
        plot_scaling(
            N_VALUES,
            results
        )

        # --------------------------------------------------------------
        # Riepilogo
        # --------------------------------------------------------------

        print(
            "\n  Riepilogo (passi medi):"
        )

        print(
            f"  {'n':>5}  "
            f"{'WS':>10}  "
            f"{'Kleinberg':>10}  "
            f"{'n^2/3':>10}  "
            f"{'log²n':>8}"
        )

        for i, n in enumerate(N_VALUES):

            sw = results["ws"][i]["steps_mean"]

            kl = results["kl"][i]["steps_mean"]

            print(
                f"  {n:>5}  "
                f"{sw:>10.1f}  "
                f"{kl:>10.1f}  "
                f"{n**(2/3):>10.1f}  "
                f"{math.log2(n)**2:>8.1f}"
            )

    except NotImplementedError as e:

        print(
            f"  [TODO] {e}"
        )

    # ==================================================================
    # BONUS
    # ==================================================================

    print(
        "\n[3] BONUS: variazione dell'esponente α"
    )

    try:

        plot_alpha_sweep(
            n=20,
            seed=42
        )

    except NotImplementedError as e:

        print(
            f"  [TODO] {e}"
        )

    print(
        "\nEsercizio completato."
    )
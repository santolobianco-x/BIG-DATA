import csv
from pathlib import Path



def export_rows(rows, output_path, overwrite=True):
    """Esporta una lista di dizionari in un file CSV.

    Se il file esiste già, non lo modifica a meno che
    overwrite=True.
    """
    rows = list(rows)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise ValueError(
            f"Nessun risultato da esportare: {output_path}"
        )

    if output_path.exists() and not overwrite:
        print(
            f"Risultati già presenti, esportazione saltata: "
            f"{output_path}"
        )
        return False

    fieldnames = list(rows[0].keys())

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Risultati salvati in: {output_path}")
    return True



def flatten_degree_bias_results(results):
    rows = []

    for baseline, quartiles in results.items():
        for quartile, metrics in quartiles.items():
            rows.append({
                "baseline": baseline,
                "quartile": quartile,
                "positive_edges": metrics["positive_edges"],
                "negative_edges": metrics["negative_edges"],
                "auc": metrics["auc"],
                "ap": metrics["ap"],
            })

    return rows

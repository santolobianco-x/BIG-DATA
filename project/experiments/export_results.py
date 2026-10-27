"""
MODULO DEDICATO ALL'ESPORTAZIONE DEI RISULTATI SPERIMENTALI
--> gestisce il salvataggio strutturato dei dati su dico sotto forma di file CSV
"""
import csv
from pathlib import Path
from config import DATASET_NAME


def export_rows(rows, output_path, overwrite=True):
    rows = list(rows)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise ValueError(
            f"No results to export: {output_path}"
        )

    if output_path.exists() and not overwrite:
        print(
            f"Results already present, export skipped: "
            f"{output_path}"
        )
        return False

    fieldnames = list(rows[0].keys())

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        # mappa automaticamente i dizionari di Python alle colonne del CSV
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader() # scrive la prima riga con i nomi delle colonne
        writer.writerows(rows)

    print(f"Results saved to: {output_path}")
    return True



def flatten_degree_bias_results(results):
    """
    Appiattisce un dizionario annidato di risultati in una lista di dizionari.
    
    I risultati dell'analisi degree-bias sono restituiti in un formato annidato:
    results[nome_baseline][quartile] = metriche

    Questa funzione converte tale struttura in un formato tabulare ideale per esportazioni.
    """
    rows = []

    for baseline, quartiles in results.items():
        for quartile, metrics in quartiles.items():
            rows.append({
                "dataset": DATASET_NAME,
                "baseline": baseline,
                "quartile": quartile,
                "positive_edges": metrics["positive_edges"],
                "negative_edges": metrics["negative_edges"],
                "auc": metrics["auc"],
                "ap": metrics["ap"],
            })

    return rows #list of dict: dati appiattiti, dove ogni dict è una riga completa

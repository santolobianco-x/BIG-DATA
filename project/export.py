# DA RIMUOVERE
from pathlib import Path

# Cartella principale del progetto
PROJECT_ROOT = Path(__file__).resolve().parent

# File di output
OUTPUT_FILE = PROJECT_ROOT / "all_python_files.txt"

# Cartelle da escludere
EXCLUDED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".idea",
    ".vscode",
    "site-packages",
}

# Raccoglie tutti i file Python del progetto
python_files = sorted(
    path
    for path in PROJECT_ROOT.rglob("*.py")
    if path != Path(__file__).resolve()
    and not any(
        part in EXCLUDED_DIRS
        for part in path.relative_to(PROJECT_ROOT).parts
    )
    and OUTPUT_FILE not in path.parents
)

with OUTPUT_FILE.open("w", encoding="utf-8") as output:
    output.write("CODICE SORGENTE PYTHON DEL PROGETTO\n")
    output.write("=" * 80 + "\n\n")

    for file_path in python_files:
        relative_path = file_path.relative_to(PROJECT_ROOT)

        output.write("\n\n")
        output.write("=" * 80 + "\n")
        output.write(f"FILE: {relative_path}\n")
        output.write("=" * 80 + "\n\n")

        try:
            output.write(file_path.read_text(encoding="utf-8"))
        except UnicodeDecodeError:
            output.write("[Impossibile leggere il file come UTF-8]\n")

        output.write("\n")

print(f"File Python trovati: {len(python_files)}")
print(f"Archivio creato: {OUTPUT_FILE}")


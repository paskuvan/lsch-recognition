import csv
import os
import shutil

# ============================================================
# MIGRACIÓN: agregar normalización por escala a datos_lsch
# ============================================================
# Los CSV recolectados antes de julio 2026 solo estaban centrados
# en la muñeca. Este script los divide por la distancia
# muñeca → nudillo del medio (landmark 9), igual que la nueva
# versión de extraer_landmarks() en paso2/paso4.
#
# Es idempotente: tras migrar, ||landmark 9|| == 1, así que
# correrlo de nuevo no cambia nada.
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "datos_lsch")
BACKUP_DIR = os.path.join(BASE_DIR, "datos_lsch_backup_prenorm")


def migrar():
    if not os.path.exists(DATA_DIR):
        print(f"❌ No existe {DATA_DIR}")
        return

    # Backup (solo la primera vez)
    if not os.path.exists(BACKUP_DIR):
        shutil.copytree(DATA_DIR, BACKUP_DIR)
        print(f"📦 Backup creado en: {BACKUP_DIR}")
    else:
        print(f"📦 Backup ya existía: {BACKUP_DIR}")

    total_filas = 0
    for letra in sorted(os.listdir(DATA_DIR)):
        csv_path = os.path.join(DATA_DIR, letra, f"{letra}.csv")
        if not os.path.exists(csv_path):
            continue

        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = [[float(v) for v in row] for row in reader]

        migradas = []
        for row in rows:
            # Landmark 9 = índices 27, 28, 29 (coords relativas a la muñeca)
            sx, sy, sz = row[27], row[28], row[29]
            escala = (sx * sx + sy * sy + sz * sz) ** 0.5
            if escala > 1e-6:
                row = [v / escala for v in row]
            migradas.append(row)

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(migradas)

        total_filas += len(migradas)
        print(f"  ✅ {letra}: {len(migradas)} muestras migradas")

    print(f"\n✅ Migración completa: {total_filas} muestras.")
    print("Siguiente paso: reentrenar con paso3_entrenar_modelo.py")


if __name__ == "__main__":
    migrar()

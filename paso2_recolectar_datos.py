import cv2
import mediapipe as mp
import os
import csv
import time
import numpy as np

# ============================================================
# PASO 2: Recolectar datos de landmarks para LSCH
# ============================================================
# Este script captura los 21 landmarks de la mano para cada
# letra del abecedario dactilológico chileno y los guarda
# en archivos CSV para entrenar el modelo después.
# ============================================================

# Configuración del modelo MediaPipe
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")

# Carpeta donde se guardarán los datos
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datos_lsch")

# Abecedario dactilológico LSCH (letras estáticas para empezar)
LETRAS = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I",
    "J", "K", "L", "M", "N", "Ñ", "O", "P", "Q",
    "R", "S", "T", "U", "V", "W", "X", "Y", "Z"
]

# Cuántas muestras capturar por letra
MUESTRAS_POR_LETRA = 100

# Conexiones para dibujar la mano
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]


def crear_carpetas():
    """Crea la carpeta principal y subcarpetas por letra."""
    os.makedirs(DATA_DIR, exist_ok=True)
    for letra in LETRAS:
        os.makedirs(os.path.join(DATA_DIR, letra), exist_ok=True)
    print(f"Carpetas creadas en: {DATA_DIR}")


def extraer_landmarks(hand_landmarks):
    """Extrae las coordenadas (x, y, z) de los 21 landmarks y las normaliza."""
    coords = []
    for lm in hand_landmarks:
        coords.extend([lm.x, lm.y, lm.z])

    # Normalizar: restar la posición de la muñeca (landmark 0)
    # para que los datos no dependan de la posición en el frame
    wrist_x, wrist_y, wrist_z = coords[0], coords[1], coords[2]
    coords_norm = []
    for i in range(0, len(coords), 3):
        coords_norm.extend([
            coords[i] - wrist_x,
            coords[i + 1] - wrist_y,
            coords[i + 2] - wrist_z,
        ])

    return coords_norm


def dibujar_mano(frame, hand_landmarks):
    """Dibuja landmarks y conexiones en el frame."""
    h, w, _ = frame.shape
    for start, end in HAND_CONNECTIONS:
        x1 = int(hand_landmarks[start].x * w)
        y1 = int(hand_landmarks[start].y * h)
        x2 = int(hand_landmarks[end].x * w)
        y2 = int(hand_landmarks[end].y * h)
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    for lm in hand_landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)


def contar_muestras_existentes(letra):
    """Cuenta cuántas muestras ya existen para una letra."""
    csv_path = os.path.join(DATA_DIR, letra, f"{letra}.csv")
    if not os.path.exists(csv_path):
        return 0
    with open(csv_path, "r") as f:
        return sum(1 for _ in f) - 1  # -1 por el header


def main():
    crear_carpetas()

    # Crear header del CSV: landmark_0_x, landmark_0_y, landmark_0_z, ..., landmark_20_z
    header = []
    for i in range(21):
        header.extend([f"lm{i}_x", f"lm{i}_y", f"lm{i}_z"])

    # Opciones del detector
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # Abrir cámara - probar índice 1 primero, luego 0
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara")
        exit()

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Esperar a que la cámara se caliente
    print("Inicializando cámara...")
    time.sleep(2)
    for _ in range(30):
        cap.read()

    print("\n" + "=" * 50)
    print("  RECOLECTOR DE DATOS - LSCH")
    print("=" * 50)
    print("\nControles:")
    print("  ESPACIO  = Iniciar/pausar captura")
    print("  N        = Siguiente letra")
    print("  P        = Letra anterior")
    print("  Q        = Salir")
    print("=" * 50)

    letra_idx = 0
    capturando = False
    muestras_capturadas = 0
    frame_timestamp_ms = 0

    with HandLandmarker.create_from_options(options) as landmarker:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame_timestamp_ms += 33

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

            letra_actual = LETRAS[letra_idx]
            existentes = contar_muestras_existentes(letra_actual)
            h, w, _ = frame.shape
            mano_detectada = False

            # Dibujar landmarks si se detecta mano
            if result.hand_landmarks:
                hand_landmarks = result.hand_landmarks[0]
                mano_detectada = True
                dibujar_mano(frame, hand_landmarks)

                # Capturar datos si está en modo captura
                if capturando and (existentes + muestras_capturadas) < MUESTRAS_POR_LETRA:
                    coords = extraer_landmarks(hand_landmarks)
                    csv_path = os.path.join(DATA_DIR, letra_actual, f"{letra_actual}.csv")

                    file_exists = os.path.exists(csv_path)
                    with open(csv_path, "a", newline="") as f:
                        writer = csv.writer(f)
                        if not file_exists:
                            writer.writerow(header)
                        writer.writerow(coords)

                    muestras_capturadas += 1

                    if (existentes + muestras_capturadas) >= MUESTRAS_POR_LETRA:
                        capturando = False
                        print(f"  ✅ Letra '{letra_actual}' completada!")

            # ---- Interfaz en pantalla ----
            # Fondo semitransparente arriba
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

            # Letra actual
            cv2.putText(frame, f"Letra: {letra_actual}", (20, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

            # Progreso
            total = existentes + muestras_capturadas
            progreso = f"{total}/{MUESTRAS_POR_LETRA}"
            cv2.putText(frame, f"Muestras: {progreso}", (20, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

            # Barra de progreso
            barra_x = 350
            barra_w = 300
            porcentaje = min(total / MUESTRAS_POR_LETRA, 1.0)
            cv2.rectangle(frame, (barra_x, 65), (barra_x + barra_w, 90), (100, 100, 100), -1)
            cv2.rectangle(frame, (barra_x, 65), (barra_x + int(barra_w * porcentaje), 90), (0, 255, 0), -1)

            # Estado
            if capturando:
                cv2.putText(frame, "CAPTURANDO...", (20, 115),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            elif not mano_detectada:
                cv2.putText(frame, "Muestra tu mano", (20, 115),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                cv2.putText(frame, "Presiona ESPACIO para capturar", (20, 115),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Lista de letras abajo
            y_lista = h - 30
            for i, letra in enumerate(LETRAS):
                color = (0, 255, 0) if contar_muestras_existentes(letra) >= MUESTRAS_POR_LETRA else (150, 150, 150)
                if i == letra_idx:
                    color = (0, 255, 255)
                x_pos = 15 + i * 30
                cv2.putText(frame, letra, (x_pos, y_lista),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2 if i == letra_idx else 1)

            cv2.imshow("Recolector LSCH", frame)

            # Controles
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):  # Espacio = iniciar/pausar captura
                capturando = not capturando
                if capturando:
                    print(f"  ▶ Capturando letra '{letra_actual}'...")
                else:
                    print(f"  ⏸ Pausa")
            elif key == ord("n"):  # N = siguiente letra
                capturando = False
                muestras_capturadas = 0
                letra_idx = min(letra_idx + 1, len(LETRAS) - 1)
                print(f"\n→ Letra: {LETRAS[letra_idx]}")
            elif key == ord("p"):  # P = letra anterior
                capturando = False
                muestras_capturadas = 0
                letra_idx = max(letra_idx - 1, 0)
                print(f"\n→ Letra: {LETRAS[letra_idx]}")

    cap.release()
    cv2.destroyAllWindows()

    # Resumen final
    print("\n" + "=" * 50)
    print("  RESUMEN DE DATOS RECOLECTADOS")
    print("=" * 50)
    for letra in LETRAS:
        count = contar_muestras_existentes(letra)
        estado = "✅" if count >= MUESTRAS_POR_LETRA else "❌"
        print(f"  {estado} {letra}: {count}/{MUESTRAS_POR_LETRA} muestras")
    print("=" * 50)


if __name__ == "__main__":
    main()

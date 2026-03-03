import cv2
import mediapipe as mp
import numpy as np
import os
import json
import time
from tensorflow import keras
from collections import deque

# ============================================================
# PASO 4: Predicción en vivo de LSCH
# ============================================================
# Usa el modelo entrenado en el Paso 3 para reconocer letras
# del abecedario dactilológico chileno en tiempo real.
# ============================================================

# Rutas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "modelo_lsch", "modelo_lsch.keras")
LABELS_PATH = os.path.join(BASE_DIR, "modelo_lsch", "letras.json")
HAND_MODEL_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")

# Configuración MediaPipe
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Conexiones para dibujar
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

# Umbral mínimo de confianza para mostrar predicción
CONFIANZA_MINIMA = 0.60

# Cuántas predicciones acumular para estabilizar
BUFFER_SIZE = 10


def extraer_landmarks(hand_landmarks):
    """Extrae y normaliza las coordenadas de los 21 landmarks."""
    coords = []
    for lm in hand_landmarks:
        coords.extend([lm.x, lm.y, lm.z])

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
    """Dibuja landmarks y conexiones."""
    h, w, _ = frame.shape
    for start, end in HAND_CONNECTIONS:
        x1, y1 = int(hand_landmarks[start].x * w), int(hand_landmarks[start].y * h)
        x2, y2 = int(hand_landmarks[end].x * w), int(hand_landmarks[end].y * h)
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    for lm in hand_landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)


def main():
    # Cargar modelo y etiquetas
    print("Cargando modelo...")
    model = keras.models.load_model(MODEL_PATH)
    with open(LABELS_PATH, "r") as f:
        LETRAS = json.load(f)
    print(f"Modelo cargado: {len(LETRAS)} letras")

    # Opciones del detector de manos
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # Abrir cámara
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara")
        exit()

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Inicializando cámara...")
    time.sleep(2)
    for _ in range(30):
        cap.read()

    print("\n" + "=" * 50)
    print("  PREDICCIÓN EN VIVO - LSCH")
    print("=" * 50)
    print("  Muestra una seña frente a la cámara")
    print("  C = Limpiar palabra")
    print("  ESPACIO = Agregar espacio a la palabra")
    print("  RETROCESO = Borrar última letra")
    print("  Q = Salir")
    print("=" * 50)

    # Buffer para estabilizar predicciones
    pred_buffer = deque(maxlen=BUFFER_SIZE)
    palabra = ""
    ultima_letra = ""
    letra_estable_count = 0
    FRAMES_PARA_CONFIRMAR = 15  # Frames seguidos con la misma letra para agregarla

    frame_timestamp_ms = 0

    with HandLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame_timestamp_ms += 33
            h, w, _ = frame.shape

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

            letra_predicha = ""
            confianza = 0.0

            if result.hand_landmarks:
                hand_landmarks = result.hand_landmarks[0]
                dibujar_mano(frame, hand_landmarks)

                # Extraer landmarks y predecir
                coords = extraer_landmarks(hand_landmarks)
                input_data = np.array([coords], dtype=np.float32)
                prediction = model.predict(input_data, verbose=0)[0]

                idx = np.argmax(prediction)
                confianza = prediction[idx]

                if confianza >= CONFIANZA_MINIMA:
                    letra_predicha = LETRAS[idx]
                    pred_buffer.append(letra_predicha)
                else:
                    pred_buffer.append("")
            else:
                pred_buffer.append("")

            # Determinar letra estable (mayoría en el buffer)
            letra_estable = ""
            if pred_buffer:
                letras_validas = [l for l in pred_buffer if l != ""]
                if len(letras_validas) > BUFFER_SIZE * 0.6:
                    from collections import Counter
                    conteo = Counter(letras_validas)
                    letra_estable = conteo.most_common(1)[0][0]

            # Agregar letra a la palabra si se mantiene estable
            if letra_estable and letra_estable == ultima_letra:
                letra_estable_count += 1
                if letra_estable_count == FRAMES_PARA_CONFIRMAR:
                    if not palabra or palabra[-1] != letra_estable:
                        palabra += letra_estable
                        print(f"  Letra agregada: {letra_estable} → Palabra: {palabra}")
            else:
                letra_estable_count = 0
                ultima_letra = letra_estable

            # ---- Interfaz en pantalla ----

            # Panel superior: letra detectada
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 130), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

            if letra_estable:
                # Letra grande
                cv2.putText(frame, letra_estable, (30, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 255, 0), 5)
                # Confianza
                cv2.putText(frame, f"{confianza * 100:.0f}%", (130, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
                # Barra de progreso para confirmar
                if letra_estable_count > 0:
                    progreso = min(letra_estable_count / FRAMES_PARA_CONFIRMAR, 1.0)
                    barra_w = 200
                    cv2.rectangle(frame, (130, 70), (130 + barra_w, 90), (100, 100, 100), -1)
                    color_barra = (0, 255, 0) if progreso >= 1.0 else (0, 200, 255)
                    cv2.rectangle(frame, (130, 70), (130 + int(barra_w * progreso), 90), color_barra, -1)
                    cv2.putText(frame, "Mantenla...", (130, 115),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)
            else:
                cv2.putText(frame, "Muestra una sena", (30, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (150, 150, 150), 2)

            # Panel inferior: palabra formada
            overlay2 = frame.copy()
            cv2.rectangle(overlay2, (0, h - 80), (w, h), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay2, 0.7, frame, 0.3, 0)

            cv2.putText(frame, "Palabra:", (20, h - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1)
            display_word = palabra if palabra else "_"
            cv2.putText(frame, display_word, (140, h - 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

            cv2.imshow("LSCH - Prediccion en Vivo", frame)

            # Controles
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("c"):
                palabra = ""
                print("  Palabra borrada")
            elif key == ord(" "):
                palabra += " "
                print(f"  Espacio agregado → Palabra: '{palabra}'")
            elif key == 8 or key == 127:  # Backspace
                palabra = palabra[:-1]
                print(f"  Última letra borrada → Palabra: '{palabra}'")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nPalabra final: {palabra}")
    print("Predicción finalizada")


if __name__ == "__main__":
    main()

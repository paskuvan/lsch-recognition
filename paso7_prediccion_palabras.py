import cv2
import mediapipe as mp
import numpy as np
import os
import json
import time
from tensorflow import keras
from collections import deque

# ============================================================
# PASO 7: Predicción en vivo de palabras LSCH
# ============================================================
# Usa el modelo LSTM entrenado en el Paso 6 para reconocer
# palabras/frases en lengua de señas chilena en tiempo real.
# Acumula 30 frames de landmarks y los clasifica.
# ============================================================

# Rutas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "modelo_palabras", "modelo_palabras.keras")
LABELS_PATH = os.path.join(BASE_DIR, "modelo_palabras", "palabras.json")
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

# Parámetros
FRAMES_POR_SECUENCIA = 30
CONFIANZA_MINIMA = 0.60
COOLDOWN_SEGUNDOS = 1.0  # Pausa entre predicciones

# Nombres para mostrar en pantalla
NOMBRES_DISPLAY = {
    "hola": "HOLA",
    "gracias": "GRACIAS",
    "por_favor": "POR FAVOR",
    "si": "SÍ",
    "no": "NO",
    "bien": "BIEN",
    "mal": "MAL",
    "mas_o_menos": "MÁS O MENOS",
}


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
    print("Cargando modelo de palabras...")
    if not os.path.exists(MODEL_PATH):
        print(f"❌ No se encontró el modelo en: {MODEL_PATH}")
        print("  Ejecuta paso6_entrenar_palabras.py primero.")
        return
    model = keras.models.load_model(MODEL_PATH)

    with open(LABELS_PATH, "r") as f:
        PALABRAS = json.load(f)
    print(f"Modelo cargado: {len(PALABRAS)} palabras")

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
    print("  PREDICCIÓN DE PALABRAS - LSCH")
    print("=" * 50)
    print("\nControles:")
    print("  ESPACIO  = Iniciar captura de secuencia")
    print("  R        = Cambiar a modo continuo/manual")
    print("  C        = Limpiar historial")
    print("  Q        = Salir")
    print("=" * 50)

    frame_timestamp_ms = 0
    buffer_frames = []  # Buffer para acumular frames
    grabando = False
    modo_continuo = False  # False = manual (ESPACIO), True = automático

    # Resultado actual
    prediccion_actual = ""
    confianza_actual = 0.0
    ultima_prediccion_time = 0

    # Historial de palabras detectadas
    historial = []

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

            h, w, _ = frame.shape
            mano_detectada = False

            if result.hand_landmarks:
                hand_landmarks = result.hand_landmarks[0]
                mano_detectada = True
                dibujar_mano(frame, hand_landmarks)

                coords = extraer_landmarks(hand_landmarks)

                # Modo continuo: mantener buffer deslizante
                if modo_continuo:
                    buffer_frames.append(coords)
                    if len(buffer_frames) > FRAMES_POR_SECUENCIA:
                        buffer_frames.pop(0)

                    # Predecir cuando tenemos suficientes frames
                    ahora = time.time()
                    if (len(buffer_frames) == FRAMES_POR_SECUENCIA and
                            ahora - ultima_prediccion_time > COOLDOWN_SEGUNDOS):
                        secuencia = np.array(buffer_frames).reshape(1, FRAMES_POR_SECUENCIA, 63)
                        pred = model.predict(secuencia, verbose=0)[0]
                        idx = np.argmax(pred)
                        conf = pred[idx]

                        if conf >= CONFIANZA_MINIMA:
                            palabra = PALABRAS[idx]
                            prediccion_actual = NOMBRES_DISPLAY.get(palabra, palabra)
                            confianza_actual = conf
                            ultima_prediccion_time = ahora

                            if not historial or historial[-1] != prediccion_actual:
                                historial.append(prediccion_actual)
                                print(f"  🔤 {prediccion_actual} ({conf * 100:.0f}%)")
                        else:
                            prediccion_actual = ""
                            confianza_actual = 0.0

                # Modo manual: acumular cuando se presiona ESPACIO
                elif grabando:
                    buffer_frames.append(coords)
                    if len(buffer_frames) >= FRAMES_POR_SECUENCIA:
                        grabando = False
                        secuencia = np.array(buffer_frames).reshape(1, FRAMES_POR_SECUENCIA, 63)
                        pred = model.predict(secuencia, verbose=0)[0]
                        idx = np.argmax(pred)
                        conf = pred[idx]

                        if conf >= CONFIANZA_MINIMA:
                            palabra = PALABRAS[idx]
                            prediccion_actual = NOMBRES_DISPLAY.get(palabra, palabra)
                            confianza_actual = conf
                            historial.append(prediccion_actual)
                            print(f"  🔤 {prediccion_actual} ({conf * 100:.0f}%)")
                        else:
                            prediccion_actual = f"? ({conf * 100:.0f}%)"
                            confianza_actual = conf
                            print(f"  ❓ No se reconoció (confianza: {conf * 100:.0f}%)")
                        buffer_frames = []
            else:
                if grabando:
                    buffer_frames.append([0.0] * 63)
                    if len(buffer_frames) >= FRAMES_POR_SECUENCIA:
                        grabando = False
                        buffer_frames = []
                        prediccion_actual = "Sin mano detectada"
                        confianza_actual = 0.0

                if not modo_continuo:
                    pass
                else:
                    buffer_frames.clear()

            # ---- Interfaz en pantalla ----
            # Panel superior
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 160), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

            # Modo actual
            modo_texto = "CONTINUO" if modo_continuo else "MANUAL"
            modo_color = (0, 200, 255) if modo_continuo else (255, 200, 0)
            cv2.putText(frame, f"Modo: {modo_texto}", (w - 250, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, modo_color, 2)

            # Predicción actual
            if prediccion_actual:
                # Color basado en confianza
                if confianza_actual >= 0.8:
                    color_pred = (0, 255, 0)
                elif confianza_actual >= 0.6:
                    color_pred = (0, 255, 255)
                else:
                    color_pred = (0, 0, 255)

                cv2.putText(frame, prediccion_actual, (20, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.8, color_pred, 4)
                cv2.putText(frame, f"{confianza_actual * 100:.0f}%", (20, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_pred, 2)
            else:
                cv2.putText(frame, "---", (20, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (150, 150, 150), 3)

            # Estado de grabación
            if grabando:
                progress = len(buffer_frames) / FRAMES_POR_SECUENCIA
                cv2.rectangle(frame, (20, 105), (320, 125), (100, 100, 100), -1)
                cv2.rectangle(frame, (20, 105), (20 + int(300 * progress), 125), (0, 0, 255), -1)
                cv2.putText(frame, f"Grabando... {len(buffer_frames)}/{FRAMES_POR_SECUENCIA}",
                            (20, 148), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            elif modo_continuo and mano_detectada:
                buf_size = len(buffer_frames)
                cv2.putText(frame, f"Buffer: {buf_size}/{FRAMES_POR_SECUENCIA}",
                            (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            elif not mano_detectada:
                cv2.putText(frame, "Muestra tu mano", (20, 125),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            elif not modo_continuo:
                cv2.putText(frame, "ESPACIO = capturar sena", (20, 125),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Historial (panel inferior)
            if historial:
                overlay2 = frame.copy()
                cv2.rectangle(overlay2, (0, h - 60), (w, h), (0, 0, 0), -1)
                frame = cv2.addWeighted(overlay2, 0.6, frame, 0.4, 0)

                # Mostrar las últimas 8 palabras
                ultimas = historial[-8:]
                texto_hist = " | ".join(ultimas)
                cv2.putText(frame, texto_hist, (20, h - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, "Historial:", (20, h - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

            cv2.imshow("Prediccion Palabras LSCH", frame)

            # Controles
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" ") and not modo_continuo and not grabando:
                grabando = True
                buffer_frames = []
                print("  🔴 Grabando secuencia...")
            elif key == ord("r"):
                modo_continuo = not modo_continuo
                buffer_frames = []
                grabando = False
                modo_str = "CONTINUO" if modo_continuo else "MANUAL"
                print(f"\n  🔄 Modo: {modo_str}")
            elif key == ord("c"):
                historial.clear()
                prediccion_actual = ""
                confianza_actual = 0.0
                print("  🗑️ Historial limpiado")

    cap.release()
    cv2.destroyAllWindows()

    # Resumen
    if historial:
        print("\n" + "=" * 50)
        print("  PALABRAS DETECTADAS EN LA SESIÓN")
        print("=" * 50)
        print("  " + " → ".join(historial))
        print("=" * 50)


if __name__ == "__main__":
    main()

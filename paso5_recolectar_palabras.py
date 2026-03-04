import cv2
import mediapipe as mp
import os
import time
import numpy as np

# ============================================================
# PASO 5: Recolectar datos de secuencias para palabras LSCH
# ============================================================
# Las palabras/frases en lengua de señas son dinámicas (tienen
# movimiento). Este script captura secuencias de 30 frames de
# landmarks y las guarda como archivos .npy para entrenar un
# modelo LSTM después.
# ============================================================

# Configuración MediaPipe
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")

# Carpeta donde se guardarán los datos
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datos_palabras")

# Palabras/frases a reconocer
PALABRAS = [
    "hola",
    "gracias",
    "por_favor",
    "si",
    "no",
    "bien",
    "mal",
    "mas_o_menos",
]

# Cuántas secuencias capturar por palabra
MUESTRAS_POR_PALABRA = 30

# Cuántos frames por secuencia (~1 segundo a 30fps)
FRAMES_POR_SECUENCIA = 30

# Conexiones para dibujar la mano
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

# Nombres bonitos para mostrar en pantalla
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


def crear_carpetas():
    """Crea carpetas para cada palabra."""
    os.makedirs(DATA_DIR, exist_ok=True)
    for palabra in PALABRAS:
        os.makedirs(os.path.join(DATA_DIR, palabra), exist_ok=True)
    print(f"Carpetas creadas en: {DATA_DIR}")


def extraer_landmarks(hand_landmarks):
    """Extrae y normaliza las coordenadas de los 21 landmarks."""
    coords = []
    for lm in hand_landmarks:
        coords.extend([lm.x, lm.y, lm.z])

    # Normalizar respecto a la muñeca (landmark 0)
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


def contar_muestras_existentes(palabra):
    """Cuenta cuántas secuencias (.npy) ya existen para una palabra."""
    carpeta = os.path.join(DATA_DIR, palabra)
    if not os.path.exists(carpeta):
        return 0
    return len([f for f in os.listdir(carpeta) if f.endswith(".npy")])


def main():
    crear_carpetas()

    # Opciones del detector
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
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
    print("  RECOLECTOR DE PALABRAS - LSCH")
    print("=" * 50)
    print("\nControles:")
    print("  ESPACIO  = Iniciar grabación de secuencia")
    print("  N        = Siguiente palabra")
    print("  P        = Palabra anterior")
    print("  Q        = Salir")
    print("\nCada grabación captura 30 frames (~1 segundo).")
    print("Realizá la seña completa después de presionar ESPACIO.")
    print("=" * 50)

    palabra_idx = 0
    grabando = False
    secuencia_actual = []  # Frames de la secuencia actual
    frame_timestamp_ms = 0
    cuenta_regresiva = 0  # Countdown antes de grabar
    cuenta_inicio = 0

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

            palabra_actual = PALABRAS[palabra_idx]
            nombre_display = NOMBRES_DISPLAY.get(palabra_actual, palabra_actual)
            existentes = contar_muestras_existentes(palabra_actual)
            h, w, _ = frame.shape
            mano_detectada = False

            # Detectar mano
            if result.hand_landmarks:
                hand_landmarks = result.hand_landmarks[0]
                mano_detectada = True
                dibujar_mano(frame, hand_landmarks)

                # Si estamos en cuenta regresiva
                if cuenta_regresiva > 0:
                    elapsed = time.time() - cuenta_inicio
                    restante = cuenta_regresiva - elapsed
                    if restante <= 0:
                        # Empezar a grabar
                        grabando = True
                        secuencia_actual = []
                        cuenta_regresiva = 0
                        print(f"  🔴 ¡GRABANDO!")

                # Grabar frames de la secuencia
                if grabando:
                    coords = extraer_landmarks(hand_landmarks)
                    secuencia_actual.append(coords)

                    if len(secuencia_actual) >= FRAMES_POR_SECUENCIA:
                        # Guardar secuencia
                        grabando = False
                        seq_array = np.array(secuencia_actual)  # (30, 63)
                        num_existentes = contar_muestras_existentes(palabra_actual)
                        filename = f"seq_{num_existentes:04d}.npy"
                        filepath = os.path.join(DATA_DIR, palabra_actual, filename)
                        np.save(filepath, seq_array)
                        print(f"  ✅ Secuencia guardada: {filename} ({num_existentes + 1}/{MUESTRAS_POR_PALABRA})")
                        secuencia_actual = []

            else:
                # Si no se detecta mano durante grabación, rellenar con ceros
                if grabando:
                    secuencia_actual.append([0.0] * 63)
                    if len(secuencia_actual) >= FRAMES_POR_SECUENCIA:
                        grabando = False
                        seq_array = np.array(secuencia_actual)
                        num_existentes = contar_muestras_existentes(palabra_actual)
                        filename = f"seq_{num_existentes:04d}.npy"
                        filepath = os.path.join(DATA_DIR, palabra_actual, filename)
                        np.save(filepath, seq_array)
                        print(f"  ⚠️ Secuencia guardada (con frames vacíos): {filename}")
                        secuencia_actual = []

            # ---- Interfaz en pantalla ----
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 140), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

            # Palabra actual
            cv2.putText(frame, f"Palabra: {nombre_display}", (20, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

            # Progreso
            total = existentes
            progreso = f"{total}/{MUESTRAS_POR_PALABRA}"
            cv2.putText(frame, f"Secuencias: {progreso}", (20, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

            # Barra de progreso
            barra_x = 380
            barra_w = 300
            porcentaje = min(total / MUESTRAS_POR_PALABRA, 1.0)
            cv2.rectangle(frame, (barra_x, 65), (barra_x + barra_w, 90), (100, 100, 100), -1)
            cv2.rectangle(frame, (barra_x, 65), (barra_x + int(barra_w * porcentaje), 90), (0, 255, 0), -1)

            # Estado
            if cuenta_regresiva > 0:
                restante = max(0, cuenta_regresiva - (time.time() - cuenta_inicio))
                cv2.putText(frame, f"Preparate... {int(restante) + 1}", (20, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
            elif grabando:
                frames_restantes = FRAMES_POR_SECUENCIA - len(secuencia_actual)
                # Barra de grabación
                grab_pct = len(secuencia_actual) / FRAMES_POR_SECUENCIA
                cv2.rectangle(frame, (20, 105), (20 + 300, 125), (100, 100, 100), -1)
                cv2.rectangle(frame, (20, 105), (20 + int(300 * grab_pct), 125), (0, 0, 255), -1)
                cv2.putText(frame, f"GRABANDO... {len(secuencia_actual)}/{FRAMES_POR_SECUENCIA}",
                            (330, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            elif not mano_detectada:
                cv2.putText(frame, "Muestra tu mano", (20, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                cv2.putText(frame, "ESPACIO = grabar secuencia", (20, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Lista de palabras abajo
            y_lista = h - 20
            x_pos = 15
            for i, palabra in enumerate(PALABRAS):
                nombre = NOMBRES_DISPLAY.get(palabra, palabra)
                color = (0, 255, 0) if contar_muestras_existentes(palabra) >= MUESTRAS_POR_PALABRA else (150, 150, 150)
                if i == palabra_idx:
                    color = (0, 255, 255)
                cv2.putText(frame, nombre, (x_pos, y_lista),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2 if i == palabra_idx else 1)
                # Calcular ancho del texto para espaciar
                (text_w, _), _ = cv2.getTextSize(nombre, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                x_pos += text_w + 20

            cv2.imshow("Recolector Palabras LSCH", frame)

            # Controles
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" ") and not grabando and cuenta_regresiva == 0:
                if existentes >= MUESTRAS_POR_PALABRA:
                    print(f"  ⚠️ '{nombre_display}' ya tiene {MUESTRAS_POR_PALABRA} muestras")
                else:
                    # Cuenta regresiva de 3 segundos
                    cuenta_regresiva = 3
                    cuenta_inicio = time.time()
                    print(f"  ⏳ Preparate para '{nombre_display}'... 3 segundos")
            elif key == ord("n"):
                grabando = False
                secuencia_actual = []
                cuenta_regresiva = 0
                palabra_idx = min(palabra_idx + 1, len(PALABRAS) - 1)
                print(f"\n→ Palabra: {NOMBRES_DISPLAY.get(PALABRAS[palabra_idx], PALABRAS[palabra_idx])}")
            elif key == ord("p"):
                grabando = False
                secuencia_actual = []
                cuenta_regresiva = 0
                palabra_idx = max(palabra_idx - 1, 0)
                print(f"\n→ Palabra: {NOMBRES_DISPLAY.get(PALABRAS[palabra_idx], PALABRAS[palabra_idx])}")

    cap.release()
    cv2.destroyAllWindows()

    # Resumen final
    print("\n" + "=" * 50)
    print("  RESUMEN DE DATOS RECOLECTADOS (PALABRAS)")
    print("=" * 50)
    for palabra in PALABRAS:
        count = contar_muestras_existentes(palabra)
        nombre = NOMBRES_DISPLAY.get(palabra, palabra)
        estado = "✅" if count >= MUESTRAS_POR_PALABRA else "❌"
        print(f"  {estado} {nombre}: {count}/{MUESTRAS_POR_PALABRA} secuencias")
    print("=" * 50)


if __name__ == "__main__":
    main()

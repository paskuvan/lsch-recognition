import cv2
import mediapipe as mp
import numpy as np
import os
import json
import time
from tensorflow import keras

from extraccion_holistica import (
    NUM_FEATURES,
    crear_detectores,
    dibujar_deteccion,
    extraer_features,
)

# ============================================================
# PASO 7: Predicción en vivo de palabras LSCH
# ============================================================
# Usa el modelo LSTM entrenado en el Paso 6 para reconocer
# palabras/frases en lengua de señas chilena en tiempo real.
# Acumula 30 frames de features (pose + 2 manos) y clasifica.
# ============================================================

# Rutas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "modelo_palabras", "modelo_palabras.keras")
LABELS_PATH = os.path.join(BASE_DIR, "modelo_palabras", "palabras.json")

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


def main():
    # Cargar modelo y etiquetas
    print("Cargando modelo de palabras...")
    if not os.path.exists(MODEL_PATH):
        print(f"❌ No se encontró el modelo en: {MODEL_PATH}")
        print("  Ejecuta paso6_entrenar_palabras.py primero.")
        return
    model = keras.models.load_model(MODEL_PATH)

    if model.input_shape[-1] != NUM_FEATURES:
        print(f"❌ El modelo espera {model.input_shape[-1]} features por frame,")
        print(f"   pero este script genera {NUM_FEATURES} (pose + 2 manos).")
        print("   Recolecta datos con paso5 y reentrena con paso6.")
        return

    with open(LABELS_PATH, "r") as f:
        PALABRAS = json.load(f)
    print(f"Modelo cargado: {len(PALABRAS)} palabras")

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
    print("  PREDICCIÓN DE PALABRAS - LSCH (manos + pose)")
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

    hand_landmarker, pose_landmarker = crear_detectores()
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame_timestamp_ms += 33

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            hand_result = hand_landmarker.detect_for_video(mp_image, frame_timestamp_ms)
            pose_result = pose_landmarker.detect_for_video(mp_image, frame_timestamp_ms)

            h, w, _ = frame.shape
            mano_detectada = bool(hand_result.hand_landmarks)

            dibujar_deteccion(frame, hand_result, pose_result)

            if mano_detectada:
                coords = extraer_features(hand_result, pose_result)

                # Modo continuo: mantener buffer deslizante
                if modo_continuo:
                    buffer_frames.append(coords)
                    if len(buffer_frames) > FRAMES_POR_SECUENCIA:
                        buffer_frames.pop(0)

                    # Predecir cuando tenemos suficientes frames
                    ahora = time.time()
                    if (len(buffer_frames) == FRAMES_POR_SECUENCIA and
                            ahora - ultima_prediccion_time > COOLDOWN_SEGUNDOS):
                        secuencia = np.array(buffer_frames).reshape(1, FRAMES_POR_SECUENCIA, NUM_FEATURES)
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
                        secuencia = np.array(buffer_frames).reshape(1, FRAMES_POR_SECUENCIA, NUM_FEATURES)
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
                    # Sin manos: igual acumular (pose puede seguir visible)
                    buffer_frames.append(extraer_features(hand_result, pose_result))
                    if len(buffer_frames) >= FRAMES_POR_SECUENCIA:
                        grabando = False
                        buffer_frames = []
                        prediccion_actual = "Sin mano detectada"
                        confianza_actual = 0.0

                # En modo continuo, sin manos se reinicia el buffer deslizante
                if modo_continuo:
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
                cv2.putText(frame, "Muestra tus manos", (20, 125),
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
    finally:
        hand_landmarker.close()
        pose_landmarker.close()

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

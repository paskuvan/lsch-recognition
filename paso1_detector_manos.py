import cv2
import mediapipe as mp
import os

# Configuración del modelo
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
DrawingUtils = mp.tasks.vision.drawing_utils
HandConnections = mp.tasks.vision.HandLandmarksConnections

# Ruta al modelo (mismo directorio que el script)
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")

# Conexiones de la mano para dibujar
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # Pulgar
    (0, 5), (5, 6), (6, 7), (7, 8),       # Índice
    (0, 9), (9, 10), (10, 11), (11, 12),  # Medio
    (0, 13), (13, 14), (14, 15), (15, 16),# Anular
    (0, 17), (17, 18), (18, 19), (19, 20),# Meñique
    (5, 9), (9, 13), (13, 17),            # Palma
]

# Crear opciones del detector
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
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

# Configurar resolución
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Esperar a que la cámara se caliente (descartar frames negros)
import time
print("Inicializando cámara...")
time.sleep(2)
for _ in range(30):
    cap.read()

print("Detector de manos en vivo - Presiona 'q' para salir")

with HandLandmarker.create_from_options(options) as landmarker:
    frame_timestamp_ms = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo leer el frame")
            break

        # Voltear imagen (espejo)
        frame = cv2.flip(frame, 1)
        frame_timestamp_ms += 33  # ~30 FPS

        # Convertir a formato MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Detectar manos
        result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        # Dibujar landmarks si se detectan manos
        h, w, _ = frame.shape
        if result.hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(result.hand_landmarks):
                # Dibujar conexiones
                for start, end in HAND_CONNECTIONS:
                    x1 = int(hand_landmarks[start].x * w)
                    y1 = int(hand_landmarks[start].y * h)
                    x2 = int(hand_landmarks[end].x * w)
                    y2 = int(hand_landmarks[end].y * h)
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Dibujar puntos
                for lm in hand_landmarks:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

                # Mostrar etiqueta (izquierda/derecha)
                if result.handedness and hand_idx < len(result.handedness):
                    label = result.handedness[hand_idx][0].category_name
                    score = result.handedness[hand_idx][0].score
                    text = f"{label} ({score:.2f})"
                    x = int(hand_landmarks[0].x * w)
                    y = int(hand_landmarks[0].y * h) + 30
                    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Mostrar el frame
        cv2.imshow("Detector de Manos", frame)

        # Salir con 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
print("Detector finalizado")

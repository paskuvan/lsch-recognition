import os

import mediapipe as mp
import numpy as np

# ============================================================
# Extracción de features "holística" para palabras LSCH
# ============================================================
# Las señas de palabras usan las dos manos y su posición
# respecto al cuerpo (ej: "gracias" parte en el mentón).
# Por eso cada frame se representa con:
#
#   pose (33 landmarks × 3)        = 99   ← dónde está cada mano
#   mano izquierda (21 × 3)        = 63   ← forma de la mano
#   mano derecha  (21 × 3)         = 63
#   ------------------------------------
#   total                          = 225 features
#
# Normalización:
#   - Pose: centrada en el punto medio de los hombros y escalada
#     por el ancho de hombros → independiente de posición/distancia.
#   - Manos: centradas en su muñeca y escaladas por la distancia
#     muñeca → nudillo del medio → solo codifican la FORMA.
#     La POSICIÓN de cada mano ya viene dada por la pose
#     (landmarks 15/16 son las muñecas).
#   - Mano no detectada → vector de ceros en su slot.
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HAND_MODEL_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")
POSE_MODEL_PATH = os.path.join(BASE_DIR, "pose_landmarker_lite.task")

POSE_FEATURES = 33 * 3
HAND_FEATURES = 21 * 3
NUM_FEATURES = POSE_FEATURES + 2 * HAND_FEATURES  # 225

# Conexiones para dibujar la mano
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

# Conexiones de la parte superior del cuerpo (hombros y brazos)
POSE_CONNECTIONS = [
    (11, 12),            # Hombros
    (11, 13), (13, 15),  # Brazo izquierdo
    (12, 14), (14, 16),  # Brazo derecho
]


def crear_detectores():
    """Crea los detectores de manos (2) y pose en modo VIDEO.

    Devuelve (hand_landmarker, pose_landmarker). Usar dentro de un
    try/finally o cerrar con .close().
    """
    BaseOptions = mp.tasks.BaseOptions
    vision = mp.tasks.vision

    hand_options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    pose_options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=POSE_MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return (
        vision.HandLandmarker.create_from_options(hand_options),
        vision.PoseLandmarker.create_from_options(pose_options),
    )


def _normalizar_mano(hand_landmarks):
    coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks], dtype=np.float32)
    coords -= coords[0]  # Centrar en la muñeca
    escala = np.linalg.norm(coords[9])  # Muñeca → nudillo del medio
    if escala > 1e-6:
        coords /= escala
    return coords.flatten()


def _normalizar_pose(pose_landmarks):
    coords = np.array([[lm.x, lm.y, lm.z] for lm in pose_landmarks], dtype=np.float32)
    centro = (coords[11] + coords[12]) / 2  # Punto medio de los hombros
    coords -= centro
    escala = np.linalg.norm(coords[11] - coords[12])  # Ancho de hombros
    if escala > 1e-6:
        coords /= escala
    return coords.flatten()


def extraer_features(hand_result, pose_result):
    """Convierte los resultados de ambos detectores en un vector (225,)."""
    pose_f = np.zeros(POSE_FEATURES, dtype=np.float32)
    if pose_result is not None and pose_result.pose_landmarks:
        pose_f = _normalizar_pose(pose_result.pose_landmarks[0])

    mano_izq = np.zeros(HAND_FEATURES, dtype=np.float32)
    mano_der = np.zeros(HAND_FEATURES, dtype=np.float32)
    if hand_result is not None and hand_result.hand_landmarks:
        for i, hand in enumerate(hand_result.hand_landmarks):
            label = "Right"
            if hand_result.handedness and i < len(hand_result.handedness):
                label = hand_result.handedness[i][0].category_name
            if label == "Left":
                mano_izq = _normalizar_mano(hand)
            else:
                mano_der = _normalizar_mano(hand)

    return np.concatenate([pose_f, mano_izq, mano_der])


def dibujar_deteccion(frame, hand_result, pose_result):
    """Dibuja manos y brazos detectados sobre el frame (BGR)."""
    import cv2

    h, w, _ = frame.shape

    if pose_result is not None and pose_result.pose_landmarks:
        pose = pose_result.pose_landmarks[0]
        for start, end in POSE_CONNECTIONS:
            x1, y1 = int(pose[start].x * w), int(pose[start].y * h)
            x2, y2 = int(pose[end].x * w), int(pose[end].y * h)
            cv2.line(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)

    if hand_result is not None and hand_result.hand_landmarks:
        for hand in hand_result.hand_landmarks:
            for start, end in HAND_CONNECTIONS:
                x1, y1 = int(hand[start].x * w), int(hand[start].y * h)
                x2, y2 = int(hand[end].x * w), int(hand[end].y * h)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            for lm in hand:
                cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 5, (255, 0, 0), -1)

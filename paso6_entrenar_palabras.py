import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import json

# ============================================================
# PASO 6: Entrenar modelo LSTM para palabras LSCH
# ============================================================
# Usa los datos de secuencias recolectados en el Paso 5 para
# entrenar una red LSTM que reconozca palabras/frases en
# lengua de señas chilena.
# ============================================================

# Rutas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "datos_palabras")
MODEL_DIR = os.path.join(BASE_DIR, "modelo_palabras")

# Palabras (deben coincidir con las del Paso 5)
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

FRAMES_POR_SECUENCIA = 30
FEATURES_POR_FRAME = 63  # 21 landmarks × 3 coordenadas


def cargar_datos():
    """Carga todas las secuencias .npy y las convierte en arrays."""
    X = []  # Features (secuencias de landmarks)
    y = []  # Labels (índice de palabra)

    for idx, palabra in enumerate(PALABRAS):
        carpeta = os.path.join(DATA_DIR, palabra)
        if not os.path.exists(carpeta):
            print(f"  ⚠️  No se encontraron datos para '{palabra}'")
            continue

        archivos = sorted([f for f in os.listdir(carpeta) if f.endswith(".npy")])
        count = 0
        for archivo in archivos:
            filepath = os.path.join(carpeta, archivo)
            secuencia = np.load(filepath)  # Shape: (30, 63)

            # Verificar forma correcta
            if secuencia.shape == (FRAMES_POR_SECUENCIA, FEATURES_POR_FRAME):
                X.append(secuencia)
                y.append(idx)
                count += 1
            else:
                print(f"  ⚠️  Forma incorrecta en {archivo}: {secuencia.shape}")

        print(f"  ✅ {palabra}: {count} secuencias cargadas")

    return np.array(X), np.array(y)


def crear_modelo(num_clases):
    """Crea el modelo LSTM para clasificación de secuencias."""
    model = keras.Sequential([
        # Entrada: secuencia de 30 frames, cada uno con 63 features
        keras.layers.Input(shape=(FRAMES_POR_SECUENCIA, FEATURES_POR_FRAME)),

        # Capas LSTM para capturar la dinámica temporal
        keras.layers.LSTM(64, return_sequences=True),
        keras.layers.Dropout(0.3),

        keras.layers.LSTM(128, return_sequences=True),
        keras.layers.Dropout(0.3),

        keras.layers.LSTM(64, return_sequences=False),
        keras.layers.Dropout(0.3),

        # Capas densas para clasificación
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.2),

        keras.layers.Dense(32, activation="relu"),

        # Salida: una neurona por palabra
        keras.layers.Dense(num_clases, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def main():
    print("=" * 50)
    print("  PASO 6: ENTRENAMIENTO DEL MODELO DE PALABRAS")
    print("=" * 50)

    # 1. Cargar datos
    print("\n📂 Cargando secuencias...")
    X, y = cargar_datos()

    if len(X) == 0:
        print("❌ No hay datos para entrenar. Ejecuta el Paso 5 primero.")
        return

    print(f"\n  Total: {len(X)} secuencias, {len(PALABRAS)} clases")
    print(f"  Forma de entrada: {X.shape}")

    # Verificar que haya suficientes datos por clase
    clases_presentes = np.unique(y)
    print(f"  Clases con datos: {len(clases_presentes)}/{len(PALABRAS)}")

    for idx in clases_presentes:
        count = (y == idx).sum()
        print(f"    {PALABRAS[idx]}: {count} secuencias")

    if len(clases_presentes) < 2:
        print("❌ Se necesitan al menos 2 clases con datos para entrenar.")
        return

    # 2. Dividir en entrenamiento y prueba
    print("\n📊 Dividiendo datos (80% train, 20% test)...")
    # Verificar si todas las clases tienen al menos 2 muestras para stratify
    min_por_clase = min((y == idx).sum() for idx in clases_presentes)
    if min_por_clase >= 2:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    else:
        print("  ⚠️ Algunas clases tienen pocas muestras, dividiendo sin stratify...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    print(f"  Entrenamiento: {len(X_train)} secuencias")
    print(f"  Prueba: {len(X_test)} secuencias")

    # 3. Crear modelo
    print("\n🧠 Creando modelo LSTM...")
    model = crear_modelo(len(PALABRAS))
    model.summary()

    # 4. Entrenar
    print("\n🏋️ Entrenando modelo...")
    history = model.fit(
        X_train, y_train,
        epochs=150,
        batch_size=16,
        validation_split=0.2,
        verbose=1,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=20,
                restore_best_weights=True,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=7,
                min_lr=0.00001,
            ),
        ],
    )

    # 5. Evaluar
    print("\n📈 Evaluando modelo...")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"  Precisión en test: {accuracy * 100:.1f}%")
    print(f"  Loss en test: {loss:.4f}")

    # Predicciones detalladas
    print("\n📋 Precisión por palabra:")
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    for idx, palabra in enumerate(PALABRAS):
        mask = y_test == idx
        if mask.sum() > 0:
            acc = (y_pred[mask] == idx).mean() * 100
            print(f"  {palabra}: {acc:.0f}% ({mask.sum()} muestras)")

    # 6. Guardar modelo
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "modelo_palabras.keras")
    model.save(model_path)
    print(f"\n💾 Modelo guardado en: {model_path}")

    # Guardar mapeo de palabras
    labels_path = os.path.join(MODEL_DIR, "palabras.json")
    with open(labels_path, "w") as f:
        json.dump(PALABRAS, f, ensure_ascii=False)
    print(f"  Etiquetas guardadas en: {labels_path}")

    print("\n" + "=" * 50)
    print(f"  ✅ Modelo entrenado con {accuracy * 100:.1f}% de precisión")
    print("  Siguiente paso: paso7_prediccion_palabras.py")
    print("=" * 50)


if __name__ == "__main__":
    main()

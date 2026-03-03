import os
import csv
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import json

# ============================================================
# PASO 3: Entrenar modelo de clasificación LSCH
# ============================================================
# Carga los datos recolectados en el Paso 2, entrena una red
# neuronal y guarda el modelo para usarlo en tiempo real.
# ============================================================

# Rutas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "datos_lsch")
MODEL_DIR = os.path.join(BASE_DIR, "modelo_lsch")

# Letras (deben coincidir con las del Paso 2)
LETRAS = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I",
    "J", "K", "L", "M", "N", "Ñ", "O", "P", "Q",
    "R", "S", "T", "U", "V", "W", "X", "Y", "Z"
]


def cargar_datos():
    """Carga todos los CSV y los convierte en arrays de NumPy."""
    X = []  # Features (landmarks)
    y = []  # Labels (índice de letra)

    for idx, letra in enumerate(LETRAS):
        csv_path = os.path.join(DATA_DIR, letra, f"{letra}.csv")
        if not os.path.exists(csv_path):
            print(f"  ⚠️  No se encontraron datos para '{letra}'")
            continue

        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            header = next(reader)  # Saltar header
            count = 0
            for row in reader:
                X.append([float(val) for val in row])
                y.append(idx)
                count += 1
            print(f"  ✅ {letra}: {count} muestras cargadas")

    return np.array(X), np.array(y)


def crear_modelo(num_features, num_clases):
    """Crea la red neuronal para clasificación de señas."""
    model = keras.Sequential([
        # Capa de entrada
        keras.layers.Input(shape=(num_features,)),

        # Capas ocultas
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.3),

        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.3),

        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dropout(0.2),

        # Capa de salida (una neurona por letra)
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
    print("  PASO 3: ENTRENAMIENTO DEL MODELO LSCH")
    print("=" * 50)

    # 1. Cargar datos
    print("\n📂 Cargando datos...")
    X, y = cargar_datos()
    print(f"\n  Total: {len(X)} muestras, {len(LETRAS)} clases")
    print(f"  Features por muestra: {X.shape[1]}")

    if len(X) == 0:
        print("❌ No hay datos para entrenar. Ejecuta el Paso 2 primero.")
        return

    # 2. Dividir en entrenamiento y prueba (80% / 20%)
    print("\n📊 Dividiendo datos (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Entrenamiento: {len(X_train)} muestras")
    print(f"  Prueba: {len(X_test)} muestras")

    # 3. Crear y mostrar modelo
    print("\n🧠 Creando modelo...")
    model = crear_modelo(X.shape[1], len(LETRAS))
    model.summary()

    # 4. Entrenar
    print("\n🏋️ Entrenando modelo...")
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        verbose=1,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=15,
                restore_best_weights=True,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=5,
                min_lr=0.0001,
            ),
        ],
    )

    # 5. Evaluar
    print("\n📈 Evaluando modelo...")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"  Precisión en test: {accuracy * 100:.1f}%")
    print(f"  Loss en test: {loss:.4f}")

    # Predicciones detalladas por letra
    print("\n📋 Precisión por letra:")
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    for idx, letra in enumerate(LETRAS):
        mask = y_test == idx
        if mask.sum() > 0:
            acc = (y_pred[mask] == idx).mean() * 100
            print(f"  {letra}: {acc:.0f}% ({mask.sum()} muestras)")

    # 6. Guardar modelo
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "modelo_lsch.keras")
    model.save(model_path)
    print(f"\n💾 Modelo guardado en: {model_path}")

    # Guardar mapeo de letras
    labels_path = os.path.join(MODEL_DIR, "letras.json")
    with open(labels_path, "w") as f:
        json.dump(LETRAS, f)
    print(f"  Etiquetas guardadas en: {labels_path}")

    print("\n" + "=" * 50)
    print(f"  ✅ Modelo entrenado con {accuracy * 100:.1f}% de precisión")
    print("  Siguiente paso: paso4_prediccion_vivo.py")
    print("=" * 50)


if __name__ == "__main__":
    main()

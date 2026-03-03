# 🤟 LSCH Recognition  
### Reconocimiento de Lengua de Señas Chilena con Inteligencia Artificial

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange?logo=tensorflow)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Hand%20Tracking-green)
![Roff](https://img.shields.io/badge/Roff-Documentation-lightgrey)
![Shell](https://img.shields.io/badge/Shell-Scripting-black?logo=gnu-bash)
![PowerShell](https://img.shields.io/badge/PowerShell-Scripting-blue?logo=powershell)

---

## 📌 Descripción

**LSCH Recognition** es un proyecto personal enfocado en el reconocimiento automático de la Lengua de Señas Chilena (LSCH) utilizando técnicas de Computer Vision y Deep Learning.

El propósito es desarrollar soluciones tecnológicas inclusivas que permitan mejorar la comunicación entre personas sordas y oyentes mediante inteligencia artificial.

---

## 🚀 Tecnologías utilizadas

- Python  
- TensorFlow  
- MediaPipe  
- Shell  
- PowerShell  
- Roff  

---

## 🧠 ¿Cómo funciona?

1. Captura de video en tiempo real.
2. Detección de manos usando MediaPipe.
3. Extracción de landmarks (puntos clave de la mano).
4. Procesamiento y clasificación con TensorFlow.
5. Conversión de la seña reconocida a texto.

---

## 📂 Estructura del proyecto

```bash
lsch-recognition/
│
├── data/               # Dataset de señas
├── models/             # Modelos entrenados
├── scripts/            # Scripts de entrenamiento y ejecución
├── docs/               # Documentación técnica (Roff)
├── main.py             # Script principal
└── README.md
```

---

## ⚙️ Instalación

```bash
# Clonar el repositorio
git clone https://github.com/paskuvan/lsch-recognition.git

# Entrar al proyecto
cd lsch-recognition

# Crear entorno virtual (opcional pero recomendado)
python -m venv venv

# Activar entorno (Mac/Linux)
source venv/bin/activate

# Activar entorno (Windows PowerShell)
venv\Scripts\Activate.ps1

# Instalar dependencias
pip install -r requirements.txt
```

---

## ▶️ Uso

```bash
python main.py
```

El sistema activará la cámara y comenzará a detectar señas en tiempo real.

---

## 🎯 Objetivo

Este proyecto busca:

- Fomentar la inclusión digital en Chile.
- Explorar el reconocimiento automático de LSCH.
- Construir la base para futuros traductores LSCH ↔ Texto.
- Aplicar IA a problemas reales de accesibilidad.

---

## 🔮 Futuras mejoras

- [ ] Aumentar precisión del modelo
- [ ] Agregar más vocabulario LSCH
- [ ] Implementar reconocimiento de frases
- [ ] Crear interfaz gráfica
- [ ] Exportar modelo a aplicación móvil

---

## 📜 Licencia

Proyecto personal con fines educativos y de investigación.

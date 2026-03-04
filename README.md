# ⚽ World Cup 2026: AI Virtual Experience

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Latest-orange.svg)

¡Vive la pasión del **Mundial 2026** con Realidad Aumentada! Este proyecto es un probador de ropa virtual (Virtual Try-On) de alto rendimiento que permite a los fans vestirse con los jerseys de sus selecciones favoritas usando solo gestos y una cámara web. 

---

## 🌟 Características Principales

* **⚡ Rendimiento Turbo:** Superposición de imágenes vectorizada con **NumPy** para garantizar una experiencia fluida (30+ FPS) incluso en resoluciones HD.
* **🖐️ Control por Gestos:** Cambia de equipación en tiempo real levantando dedos (Ej: 1 dedo para Argentina, 2 para Portugal).
* **👥 Soporte Multijugador:** Capacidad de procesar hasta **4 personas simultáneamente** en la misma toma.
* **🧠 HUD Inteligente:** Interfaz flotante sobre la cabeza de cada usuario que se ajusta dinámicamente según la distancia a la cámara.
* **👕 Ajuste Automático:** El jersey detecta la inclinación de los hombros y el ancho del torso para un calce realista.

---

## 🛠️ Tecnologías Utilizadas

* **Python 3.x**
* **OpenCV:** Manipulación de video y renderizado de interfaz.
* **MediaPipe (Pose & Hands):** Modelos de IA para el seguimiento del cuerpo y reconocimiento de gestos.
* **NumPy:** Procesamiento matricial de alto rendimiento para la mezcla de canales alfa.

---

## 🚀 Instalación y Uso

1. **Clona el repositorio:**
   ```bash
   git clone [https://github.com/tu-usuario/world-cup-2026-ai-swapper.git](https://github.com/tu-usuario/world-cup-2026-ai-swapper.git)
   cd world-cup-2026-ai-swapper
 
2. **Instala las dependencias:**
   ```bash
   
    pip install opencv-python mediapipe numpy
   
3. **Descarga los modelos de MediaPipe:**
Asegúrate de tener los archivos .task en la raíz del proyecto:
* pose_landmarker_heavy.task
* hand_landmarker.task

## 📂 Estructura de Archivos
```
📁 world-cup-2026-ai-swapper/
│
├── 📄 main.py                   # Script principal de ejecución
├── 📄 pose_landmarker_heavy.task # Modelo de Pose MediaPipe
├── 📄 hand_landmarker.task      # Modelo de Manos MediaPipe
└── 📁 Body/                     # Carpeta de assets gráficos
    ├── 🖼️ Camisetas_Mundial_arg.png
    └── 🖼️ Camisetas_Mundial_port.png
```

# YOLOv8 Pipelines: Preprocesamiento y Entrenamiento Modular

Este proyecto implementa un flujo completo para entrenamiento de modelos de detección de objetos con YOLOv8, incluyendo preparación de datos, cálculo de estadísticas y pipelines de entrenamiento configurables.

---

## 🚀 Objetivo

Construir una base modular y reproducible para experimentación con modelos YOLOv8, permitiendo extender fácilmente el preprocesamiento y comparar distintas estrategias de entrenamiento.

---

## 🧠 Enfoque

El proyecto está estructurado en pipelines independientes que comparten el mismo dataset pero difieren en el preprocesamiento aplicado.

Actualmente se incluyen:

### 🔹 Pipeline A (Base)
- Conversión de PASCAL VOC a formato YOLO
- Entrenamiento estándar con YOLOv8

### 🔹 Pipeline B (Extendido)
- Uso de estadísticas por canal (mean/std)
- Estandarización de imágenes durante entrenamiento y validación
- Integración mediante trainer personalizado

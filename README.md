# Clasificación de neumonía en radiografías de tórax
### Proyecto semestral de Visión Computacional · Entregable 1: Preprocesamiento

*Universidad:* Universidad Nacional de San Antonio Abad del Cusco (UNSAAC)
*Curso:* Visión Computacional
*Integrantes:* (completar, máximo 3)

---

## 1. Dataset

[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) (Kaggle): 5 856 radiografías de tórax pediátricas con dos clases, NORMAL y PNEUMONIA.

*¿Por qué es un dataset desafiante?*

| Problema | Evidencia |
|---|---|
| Desbalance de clases | 73 % neumonía (4 273) vs 27 % normal (1 583) |
| Tamaños muy variables | De 127 a 2 916 px; proporción ancho/alto entre 0.84 y 3.38 |
| Formatos mezclados | 5 573 en escala de grises y 283 en RGB (todas de neumonía) |
| Duplicados | 62 imágenes duplicadas exactas (30 grupos) |
| Split original poco confiable | El conjunto de validación tiene solo 16 imágenes |
| Riesgo de fuga de datos | Un mismo paciente puede tener varias imágenes |

## 2. Estructura del repositorio


├── preprocesamiento_chest_xray.ipynb   # Notebook completo (ejecutar en Google Colab)
├── manifiesto.csv                      # Split final por paciente y ruta de cada imagen procesada
├── requirements.txt                    # Dependencias
└── README.md


> Las imágenes procesadas *no se suben al repositorio* por su tamaño. Se regeneran ejecutando el notebook.

## 3. Cómo reproducir

1. Abrir preprocesamiento_chest_xray.ipynb en [Google Colab](https://colab.research.google.com/).
2. Ejecutar todas las celdas (Entorno de ejecución → Ejecutar todo). El dataset se descarga automáticamente con kagglehub (~2 GB).
3. Las imágenes procesadas quedan en dataset_preprocesado/{train,val,test}/{NORMAL,PNEUMONIA}/.

Para ejecutar en local: pip install -r requirements.txt.

## 4. Metodología

### 4.1 Exploración y control de calidad
- Conteo por clase y split, tamaños, modos de color y muestras visuales.
- *Imágenes corruptas:* 0.
- *Imágenes RGB:* las 283 tienen los tres canales idénticos, es decir, son grises.
- *Imágenes anómalas* (alto < 300 px o proporción > 2): 150.
- *Duplicados:* se detectaron con hash MD5 (exactos) y con hash perceptual (dHash). La versión 8×8 daba falsos positivos (mezclaba clases); con 16×16 coincide con MD5, por lo que solo se eliminan los exactos.

### 4.2 Limpieza
5 856 → 5 824 (sin duplicados exactos) → *5 674* (sin anómalas).

### 4.3 División por paciente
Nuevo split 70/15/15 con StratifiedGroupKFold, agrupando por paciente (identificado en el nombre del archivo) y estratificado por clase. Se verificó que *ningún paciente aparece en más de un conjunto*.

| Split | NORMAL | PNEUMONIA | Total | % neumonía |
|---|---|---|---|---|
| Train | 1 132 | 2 904 | 4 036 | 72.0 |
| Val | 230 | 552 | 782 | 70.6 |
| Test | 215 | 641 | 856 | 74.9 |
| *Total* | 1 577 | 4 097 | 5 674 | 72.2 |

### 4.4 Pipeline de preprocesamiento
Aplicado a las 5 674 imágenes y guardado como PNG:

1. *Escala de grises* (un solo canal).
2. *Redimensionado a 224×224* conservando la proporción, con relleno negro (sin deformar).
3. *Suavizado gaussiano 3×3* para reducir ruido.
4. *CLAHE* (clipLimit=2.0, tiles 8×8) para mejorar el contraste local.

Verificación: las 5 674 imágenes son 224×224, uint8 y de un canal.

### 4.5 Preparación para el entrenamiento
- *Normalización* con media y desviación calculadas solo con train: media = 0.3793, desviación = 0.3104.
- *Pesos de clase:* NORMAL = 1.783, PNEUMONIA = 0.695 (no se rebalancea el dataset).
- *Etiquetas:* NORMAL = 0, PNEUMONIA = 1.
- *Data augmentation* (solo train, durante el entrenamiento): rotación ±10°, zoom 0.9-1.1 y brillo/contraste leves. No se usa volteo horizontal porque el corazón está siempre del mismo lado.

La función cargar_split(split) del notebook devuelve las imágenes normalizadas, con forma (N, 224, 224, 1) en float32, listas para el modelo.

## 5. Limitaciones

- Las imágenes NORMAL son en promedio más grandes que las de neumonía, y varias traen marcadores de texto (como una "R"). El modelo podría aprender esos rasgos en lugar de la patología. El redimensionado con relleno reduce ese riesgo, pero no lo elimina.
- El identificador de paciente se extrae del nombre del archivo; si dos pacientes comparten número, podrían agruparse juntos.

## 6. Próximos pasos
Entrenar y evaluar modelos de clasificación sobre este dataset preprocesado, usando los pesos de clase y priorizando métricas como recall, F1 y AUC por el desbalance.

## 7. Tecnologías
Python · OpenCV · NumPy · pandas · scikit-learn · Matplotlib · Seaborn · Google Colab

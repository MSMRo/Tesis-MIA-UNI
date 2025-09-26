# 🧠 Generación Sintética de Señales ECG para Aplicaciones en Educación y Validación de Algoritmos

## 1️⃣ Contexto y Motivación

En el campo de la **Ingeniería Biomédica** y áreas afines, existe una **carencia de bases de datos de señales ECG accesibles, amplias y balanceadas** que representen adecuadamente la variabilidad fisiológica y patológica. Las bases públicas existentes, como **PTB-XL** y **MIT-BIH Arrhythmia Database (MITDB) de [PhysioNet](https://physionet.org/files/mitdb)**, aunque valiosas, presentan limitaciones:

* Tamaños de muestra relativamente pequeños para ciertas patologías.
* Variabilidad restringida en poblaciones y condiciones.
* Anotaciones heterogéneas o insuficientes para algunos fines educativos e investigativos.

Esto genera dificultades en:

* **Formación académica**, al impedir que estudiantes practiquen con datos diversos y realistas.
* **Investigación y validación de algoritmos**, por falta de datos suficientes para entrenar modelos robustos.
* **Reproducibilidad científica**, debido a restricciones de licencias o tamaños limitados.

La propuesta busca **desarrollar un modelo generativo basado en GANs (Generative Adversarial Networks)** para producir señales ECG sintéticas realistas y parametrizables, complementando y ampliando bases como **PTB-XL** y **MITDB**, facilitando la docencia y la investigación.

---

## 2️⃣ Objetivo General

Evaluar el desempeño de un **modelo generativo de señales ECG basado en GANs**, midiendo la similitud morfológica y temporal de los complejos **P-QRS-T** respecto a señales reales provenientes de bases de datos públicas de referencia (**PTB-XL y MIT-BIH Arrhythmia Database**).

### Objetivos específicos

* Entrenar y ajustar un **modelo GAN** para generar señales ECG con variabilidad controlada (frecuencia cardíaca, morfología, alteraciones comunes).
* Establecer un conjunto de métricas objetivas para comparar señales sintéticas con señales reales.
* Validar la capacidad del modelo para preservar características fisiológicas clave (duración y amplitud de ondas, intervalos PR, QT, RR).
* Generar un conjunto de datos sintético documentado y reproducible para uso educativo y validación de algoritmos.

---

## 3️⃣ Metodología

**Etapas:**

1. **Revisión bibliográfica**

   * Modelos matemáticos de ECG (McSharry et al.).
   * Arquitecturas GAN aplicadas a datos biomédicos (cGAN, WGAN, TimeGAN).

2. **Preparación de datos**

   * Selección y limpieza de datasets públicos: **PTB-XL** y **MIT-BIH Arrhythmia Database (MITDB)**.
   * Normalización de amplitud y frecuencia de muestreo.
   * Anotación de complejos P, QRS y T.

3. **Diseño y entrenamiento del modelo**

   * Arquitecturas candidatas: WGAN-GP, TimeGAN, cGAN condicional en ritmo y frecuencia.
   * Evaluación iterativa de estabilidad de entrenamiento y calidad de señales.

4. **Evaluación y validación**

   * Comparación cuantitativa y cualitativa de las señales generadas vs. reales usando métricas objetivas y análisis visual.

---

## 4️⃣ Avances Técnicos

### 🔹 Generación Sintética de ECG (`GEN_EKG.ipynb`)

* Implementación de **generadores basados en arquitecturas recurrentes y convolucionales** para capturar dependencias temporales y morfología ECG.
* Configuración inicial de WGAN y TimeGAN para generación de segmentos de latido.
* Visualización de señales sintéticas y comparación inicial de complejos P-QRS-T con datos reales.

### 🔹 Exploración y Análisis de Datos (`EDA_dataset.ipynb`)

* Limpieza y balanceo de datasets **PTB-XL** y **MITDB**.
* Extracción de características temporales y amplitud de P-QRS-T.
* Primeras comparaciones estadísticas entre latidos reales y generados.

---

## 5️⃣ Métricas recomendadas para evaluar similitud ECG

**Dominio señal / morfología**

* **RMSE (Root Mean Square Error)** y **MAE (Mean Absolute Error)**: cuantifican diferencia punto a punto.
* **CC / Pearson Correlation Coefficient**: mide correlación global entre señales.
* **Dynamic Time Warping (DTW) distance**: robusto a ligeros desajustes temporales entre señales.
* **FID adaptado (Fréchet Inception Distance modificado para series temporales)**: evalúa similitud en espacio latente.

**Dominio clínico / eventos**

* **Error porcentual de amplitud y tiempo de picos P, QRS, T**.
* **ΔRR y HRV (Heart Rate Variability)**: consistencia en variabilidad de intervalos RR.
* **Waveform Similarity Index (WSI)** o **Normalized Cross-Correlation (NCC)**: útil para forma de onda.

**Dominio frecuencia**

* **PSD (Power Spectral Density) similarity**: comparar distribución de energía en bandas relevantes.

> **Recomendación práctica:**
> Combinar métricas generales (RMSE, DTW, Pearson) con métricas clínicas (error de picos P-QRS-T) y FID adaptado para una validación robusta y multidimensional.

---

## 6️⃣ Impacto y Alcance

* **Académico**: democratiza la enseñanza práctica de bioseñales y generación de datasets sintéticos confiables.
* **Investigación**: posibilita probar algoritmos de clasificación y detección de arritmias sin depender solo de datos reales.
* **Tecnológico**: promueve el uso de **modelos generativos avanzados (GANs)** en biomedicina.
* **Escalabilidad**: adaptable a otras señales fisiológicas (EMG, EEG) y nuevos modelos generativos.

---

## 7️⃣ Próximos Pasos

* Mejorar estabilidad y realismo de la GAN con WGAN-GP y regularización espectral.
* Calcular métricas combinadas (RMSE, DTW, correlación, FID) sobre dataset de validación.
* Generar un conjunto curado de señales sintéticas etiquetadas con sus parámetros fisiológicos.
* Documentar el pipeline para publicación y uso educativo.




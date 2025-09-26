# 🧠 Desarrollo de un Sistema Integrado de Simulación de Señales ECG
## 1️⃣ Contexto y Motivación

En cursos de Procesamiento de Señales Médicas (PDSM) para Ingeniería Biomédica y áreas afines, la etapa de adquisición real de señales suele omitirse o sustituirse por simuladores comerciales de alto costo (p. ej., PROSIM de FLUKE). Esto:

Limita la comprensión del flujo completo de señal (generación → adquisición → análisis).

Aumenta la barrera económica para equipar laboratorios docentes.

La propuesta busca un sistema económico, reproducible y escalable que integre generación sintética de ECG mediante LLM, hardware abierto y adquisición real para mejorar la formación práctica.

## 2️⃣ Objetivo General

Evaluar el desempeño de un sistema integrado para generación y adquisición de señales ECG, controlado mediante instrucciones en lenguaje natural y un modelo de lenguaje entrenado (LLM), midiendo el error porcentual medio de amplitud y posición temporal de los complejos P-QRS-T frente a un simulador comercial de referencia.

Objetivos específicos

Afinar y configurar un LLM para generar formas de onda ECG parametrizadas.

Construir hardware con DAC y acondicionamiento de señal compatible con equipos médicos.

Diseñar protocolos de prueba para comparar señales con un simulador comercial.

Evaluar el error medio de amplitud y tiempo en los complejos P, QRS y T.

Realizar análisis estadístico de desempeño y usabilidad con estudiantes.

## 3️⃣ Metodología

Etapas:

Revisión bibliográfica

Modelos matemáticos de ECG (p. ej. McSharry et al.)

Aplicación de LLMs en control y generación de señales médicas.

Diseño y construcción de hardware

Microcontrolador con DAC (ESP32 / STM32).

Etapa de acondicionamiento de señal y emulación de impedancias.

Integración con Arduino para adquisición y muestreo.

Desarrollo de software

Backend en Python para generación de ECG sintético.

Interfaz gráfica (PyQt/Tkinter) con control manual y por lenguaje natural.

Comunicación bidireccional PC ↔ Arduino.

Fine-tuning del LLM

Dataset de instrucciones en lenguaje natural → parámetros ECG (frecuencia, amplitud, arritmias).

Entrenamiento y validación en plataformas como Hugging Face.

Validación

Comparación con simulador comercial (referencia).

Talleres con estudiantes para evaluar precisión y usabilidad.

## 4️⃣ Avances Técnicos
🔹 Generación Sintética de ECG (GEN_EKG.ipynb)

Modelos de onda ECG: Implementación basada en parámetros fisiológicos ajustables (frecuencia cardíaca, amplitud de P, QRS y T).

Control por lenguaje natural: Pruebas iniciales para traducir texto a parámetros de onda.

Visualización: Graficado dinámico de señales generadas y comparación con formas de onda de referencia.

🔹 Exploración y Análisis de Datos (EDA_dataset.ipynb)

Carga y limpieza de datasets ECG: Integración con bases como PTB-XL y otras públicas.

Extracción de características: picos P-QRS-T, intervalos RR, variabilidad de amplitud y tiempo.

Métricas de calidad de señal: Error porcentual en amplitud y desplazamiento temporal inicial.

## 5️⃣ Impacto y Alcance

Académico: democratiza la enseñanza práctica de bioseñales con hardware accesible.

Tecnológico: integra IA generativa (LLM) para control intuitivo y personalización de señales.

Escalabilidad: adaptable a otras bioseñales (EMG, EEG) y distintos niveles educativos.

Investigación: facilita experimentación en generación sintética y análisis de bioseñales.

## 6️⃣ Próximos Pasos

Terminar el fine-tuning del LLM con dataset curado de instrucciones y parámetros ECG.

Construir el prototipo de hardware y realizar pruebas de señal DAC vs simulador comercial.

Validar error medio (<5%) en amplitud y tiempo de P-QRS-T.

Integrar interfaz gráfica completa y pruebas de usuario con estudiantes.
# 📊 GUÍA COMPLETA DE INTERPRETACIÓN DE RESULTADOS

## 🎯 Objetivo

El VAE Condicional Mejorado genera señales ECG sintéticas que replican características de las señales originales. Esta guía te ayuda a interpretar si las señales generadas son de buena calidad.

---

## 📈 Análisis de Curvas de Entrenamiento

### 1. Total Loss (Pérdida Total)
```
Esperado: Decrecimiento suave hacia convergencia
Rango: Comienza alto (1.0-2.0), termina bajo (0.01-0.1)
```

**✅ Bueno:**
- Curva suave sin saltos abruptos
- Convergencia alrededor del epoch 500-600
- Sin oscilaciones erráticas al final

**⚠️ Problemas:**
- Curva ruidosa = learning rate muy alto
- Estancamiento temprano = learning rate muy bajo
- Oscilaciones = inestabilidad numérica

### 2. Reconstruction Loss (MSE)
```
Mide: Calidad de reconstrucción de la forma de onda
Esperado: Decrecimiento consistente
```

**✅ Bueno:**
- Cae rápidamente en primeros 100 epochs (warmup)
- Continúa cayendo gradualmente hasta epoch 700
- Termina en rango 0.005-0.05

**⚠️ Problemas:**
- Estancamiento = modelo no aprende características
- Aumento posterior = overfitting

### 3. KL Divergence Loss
```
Mide: Similitud del espacio latente con distribución normal
Esperado: Aumento gradual después del warmup
```

**✅ Bueno:**
- Cercano a 0 en primeros 100 epochs (warmup)
- Aumenta gradualmente después (KL annealing)
- Estabiliza alrededor del epoch 500

**⚠️ Problemas:**
- Aumento brusco = KL weight muy alto
- Permanece en 0 = regularización insuficiente

### 4. KL Weight Annealing
```
Mide: Evolución del peso del término KL
Esperado: Aumento lineal durante warmup
```

**✅ Bueno:**
- Comienza en 0.0
- Aumenta linealmente hasta epoch 100
- Se estabiliza en 0.05 desde epoch 100-700

---

## 🔬 Análisis de Métricas de Coherencia

### Tabla de Interpretación General

```
┌─────────────────────┬──────────┬──────────┬──────────┬──────────┐
│ Métrica             │ Excelente│  Bueno   │ Aceptable│   Pobre  │
├─────────────────────┼──────────┼──────────┼──────────┼──────────┤
│ Pearson Correlation │  > 0.80  │ 0.70-0.80│ 0.60-0.70│  < 0.60  │
│ Spectral Similarity │  > 0.75  │ 0.65-0.75│ 0.55-0.65│  < 0.55  │
│ Energy Similarity   │  > 0.85  │ 0.75-0.85│ 0.60-0.75│  < 0.60  │
│ SNR (dB)            │  > 35    │  25-35   │  15-25   │  < 15    │
│ DTW Distance        │  < 0.1   │ 0.1-0.3  │ 0.3-0.5  │  > 0.5   │
│ Frechet Distance    │  < 0.15  │ 0.15-0.35│ 0.35-0.55│  > 0.55  │
│ MSE (×10⁻³)        │  < 5     │  5-15    │  15-30   │  > 30    │
│ MAE (×10⁻³)        │  < 8     │  8-20    │  20-40   │  > 40    │
└─────────────────────┴──────────┴──────────┴──────────┴──────────┘
```

---

## 🔍 Interpretación por Métrica

### 1️⃣ Correlación de Pearson

**¿Qué mide?**
- Similitud en patrones y tendencias
- Rango: [-1, 1], donde 1 = perfecta correlación

**¿Cómo interpretarlo?**

```
0.85-1.00  ✅ EXCELENTE - Patrones casi idénticos
0.70-0.85  ✅ BUENO     - Patrones muy similares
0.60-0.70  ⚠️  ACEPTABLE - Patrones parcialmente similares
0.40-0.60  ❌ POBRE     - Patrones débilmente similares
< 0.40     ❌ MALO      - Patrones disimilares
```

**Ejemplo interpretativo:**
- NSR con Pearson 0.82: Las formas de onda sintéticas siguen patrones de ritmo cardíaco normales
- Bigeminy con Pearson 0.65: Las formas de onda sintéticas capturan parcialmente el patrón de latidos ectópicos

---

### 2️⃣ Similitud Espectral

**¿Qué mide?**
- Similitud en contenido de frecuencia (usando FFT)
- Indica si las "vibraciones" son similares

**¿Cómo interpretarlo?**

```
0.75-1.00  ✅ EXCELENTE - Espectro casi idéntico
0.65-0.75  ✅ BUENO     - Espectro muy similar
0.55-0.65  ⚠️  ACEPTABLE - Espectro parcialmente similar
0.35-0.55  ❌ POBRE     - Espectro débilmente similar
< 0.35     ❌ MALO      - Espectro disimilar
```

**Por qué importa:**
- NSR tiene frecuencias dominantes claras → Espectral debe ser alto
- Aritmias complejas tienen espectros dispersos → Espectral más bajo es aceptable

---

### 3️⃣ Similitud de Energía

**¿Qué mide?**
- Similitud en energía total = amplitud y varianza
- Rango: [0, 1], donde 1 = energía idéntica

**¿Cómo interpretarlo?**

```
0.85-1.00  ✅ EXCELENTE - Energía casi idéntica
0.75-0.85  ✅ BUENO     - Energía muy similar
0.60-0.75  ⚠️  ACEPTABLE - Energía parcialmente similar
0.40-0.60  ❌ POBRE     - Energía débilmente similar
< 0.40     ❌ MALO      - Energía disimilar
```

**Problema común:**
- Si SNR es alto pero Energy es bajo → El modelo subestima amplitudes
- Solución: Aumentar FINAL_KL_WEIGHT o reducir Dropout

---

### 4️⃣ Signal-to-Noise Ratio (SNR)

**¿Qué mide?**
- Relación entre la potencia de la señal original vs error
- Unidades: dB (decibeles)
- Fórmula: 10 × log₁₀(Potencia_señal / Potencia_error)

**¿Cómo interpretarlo?**

```
> 40 dB   ✅ EXCELENTE - Casi sin error perceptible
25-40 dB  ✅ BUENO     - Error bajo pero detectable
15-25 dB  ⚠️  ACEPTABLE - Error notable
< 15 dB   ❌ POBRE     - Error muy alto
```

**Escala práctica:**
- 6 dB ≈ 25% de error
- 12 dB ≈ 6% de error
- 20 dB ≈ 1% de error

**Interpretación para ECG:**
- SNR > 25 dB es generalmente aceptable para síntesis
- SNR > 35 dB es excelente

---

### 5️⃣ Dynamic Time Warping (DTW)

**¿Qué mide?**
- Distancia mínima entre señales permitiendo "warping" temporal
- Útil cuando timing exacto es menos importante que forma general
- Rango: [0, ∞), donde 0 = señales idénticas

**¿Cómo interpretarlo?**

```
< 0.10     ✅ EXCELENTE - Señales casi idénticas
0.10-0.30  ✅ BUENO     - Señales muy similares
0.30-0.50  ⚠️  ACEPTABLE - Señales parcialmente similares
0.50-1.00  ❌ POBRE     - Señales débilmente similares
> 1.00     ❌ MALO      - Señales muy disimilares
```

**Ventaja del DTW:**
- Captura similitud incluso si los picos están ligeramente desalineados
- Mejor que correlación simple para aritmias complejas

---

### 6️⃣ Frechet Distance

**¿Qué mide?**
- Distancia máxima punto a punto entre curvas
- Como la "brecha mayor" entre dos trayectorias
- Rango: [0, ∞), donde 0 = idénticas

**¿Cómo interpretarlo?**

```
< 0.15     ✅ EXCELENTE - Máximo gap muy pequeño
0.15-0.35  ✅ BUENO     - Máximo gap pequeño
0.35-0.55  ⚠️  ACEPTABLE - Máximo gap moderado
> 0.55     ❌ POBRE     - Máximo gap grande
```

**Interpretación:**
- Si Frechet es alto pero Pearson es alto → Hay picos desalineados pero patrón es similar
- Si ambos son altos → Señales muy diferentes

---

### 7️⃣ Mean Squared Error (MSE)

**¿Qué mide?**
- Error cuadrático promedio entre puntos
- Penaliza más los errores grandes

**¿Cómo interpretarlo?**

```
< 0.005    ✅ EXCELENTE - Error muy bajo
0.005-0.015 ✅ BUENO     - Error bajo
0.015-0.030 ⚠️  ACEPTABLE - Error moderado
> 0.030    ❌ POBRE     - Error alto
```

**Relación con SNR:**
- MSE bajo = SNR alto
- Si SNR es alto pero MSE es alto → Escala de datos grande

---

### 8️⃣ Mean Absolute Error (MAE)

**¿Qué mide?**
- Error absoluto promedio (sin penalizar grandes errores)
- Más robusto a outliers que MSE

**¿Cómo interpretarlo?**

```
< 0.008    ✅ EXCELENTE - Error muy bajo
0.008-0.020 ✅ BUENO     - Error bajo
0.020-0.040 ⚠️  ACEPTABLE - Error moderado
> 0.040    ❌ POBRE     - Error alto
```

---

## 🎯 Interpretación por Clase

### Bigeminy (Latidos Ectópicos Alternados)
**Características:**
- Patrón alternado: latido normal → latido ectópico
- Menos regular que NSR
- Amplitudes variables

**Expectativas realistas:**
- Pearson: 0.68-0.80 (patrón alternado es más difícil)
- Spectral: 0.62-0.75
- Energy: 0.70-0.85
- SNR: 14-24 dB

**Red flags:**
- Si Pearson < 0.55: No captura patrón alternado
- Si Spectral < 0.50: Contenido de frecuencia completamente diferente

---

### NSR (Normal Sinus Rhythm)
**Características:**
- Patrón muy regular y periódico
- Amplitudes constantes
- Espectro con picos claros

**Expectativas realistas:**
- Pearson: 0.75-0.90 (debe ser alta por regularidad)
- Spectral: 0.70-0.85 (espectro regular facilita síntesis)
- Energy: 0.80-0.95 (energía debe ser muy similar)
- SNR: 18-28 dB

**Red flags:**
- Si Pearson < 0.65: Regularidad no se captura
- Si Energy < 0.70: Amplitudes incorrecto
- Si NSR tiene Pearson < Bigeminy: Algo está mal

---

### Trigeminy (Latidos Ectópicos Cada 3 Latidos)
**Características:**
- Patrón cada 3 latidos: normal, normal, ectópico
- Regularidad parcial
- Amplitudes variadas

**Expectativas realistas:**
- Pearson: 0.65-0.80
- Spectral: 0.60-0.75
- Energy: 0.65-0.85
- SNR: 13-23 dB

**Red flags:**
- Si patrón no es cada 3 latidos: Modelo no aprendió estructura

---

## 📊 Combinaciones de Métricas

### Caso 1: Pearson ALTO, Spectral BAJO
```
Interpretación:
- Forma general es similar
- Pero contenido de frecuencia es diferente
- Posible: Amplitudes escaladas incorrectamente

Acción:
- Revisar si las señales están normalizadas correctamente
- Posible problema en desnormalización
```

### Caso 2: Pearson BAJO, Spectral ALTO
```
Interpretación:
- Contenido de frecuencia es similar
- Pero forma general es diferente
- Posible: Picos desalineados

Acción:
- DTW debería ser relativamente bajo
- Si DTW también es alto → Problema serio
```

### Caso 3: Energy BAJO, SNR ALTO
```
Interpretación:
- Error es pequeño pero relativo a escala baja
- Las señales tienen menos amplitud de la esperada

Acción:
- Aumentar factor de desnormalización
- Revisar X_min y X_max
```

### Caso 4: DTW ALTO, Frechet BAJO
```
Interpretación:
- Máxima diferencia es pequeña
- Pero diferencia temporal es grande
- Posible: Desalineamiento de fase

Acción:
- Revisar si existe desfase constante
- Considerar agregar regularización de fase
```

---

## ✅ Criterios de Aceptación

### SÍNTESIS EXCELENTE
```
✅ Pearson > 0.80
✅ Spectral > 0.75
✅ Energy > 0.85
✅ SNR > 30 dB
✅ DTW < 0.2
✅ MSE < 0.010
```

### SÍNTESIS BUENA
```
✅ Pearson > 0.70
✅ Spectral > 0.65
✅ Energy > 0.75
✅ SNR > 22 dB
✅ DTW < 0.35
✅ MSE < 0.020
```

### SÍNTESIS ACEPTABLE
```
✅ Pearson > 0.60
✅ Spectral > 0.55
✅ Energy > 0.60
✅ SNR > 15 dB
✅ DTW < 0.50
✅ MSE < 0.035
```

### SÍNTESIS INSUFICIENTE
```
❌ Pearson < 0.60
❌ Spectral < 0.55
❌ Energy < 0.60
❌ SNR < 15 dB
❌ DTW > 0.50
```

---

## 🔧 Acciones Correctivas

### Si Pearson es BAJO
```
Causas posibles:
1. Modelo no converge (aumentar EPOCHS)
2. Learning rate muy alto (reducir LEARNING_RATE)
3. KL weight muy alto (reducir FINAL_KL_WEIGHT)
4. Latent dimension insuficiente (aumentar LATENT_DIM)

Acciones:
- Verificar curvas de entrenamiento
- Aumentar EPOCHS a 800-900
- Reducir LEARNING_RATE a 0.0001
- Reducir FINAL_KL_WEIGHT a 0.03
```

### Si Spectral es BAJO
```
Causas posibles:
1. Ruido en generación
2. Contenido de frecuencia no aprendido
3. Desnormalización incorrecta

Acciones:
- Aumentar FINAL_KL_WEIGHT (más regularización)
- Reducir Dropout rates
- Verificar que FFT se calcula correctamente
```

### Si Energy es BAJO
```
Causas posibles:
1. Desnormalización incorrecta
2. X_min/X_max no calculados correctamente
3. Amplitudes subestimadas por modelo

Acciones:
- Verificar cálculo: signal = (signal_norm + 1) / 2 * (X_max - X_min) + X_min
- Aumentar factor de escala
- Reducir KL_WEIGHT (permite mayor varianza)
```

### Si SNR es BAJO
```
Causas posibles:
1. Gran error de reconstrucción general
2. Modelo no aprende bien
3. Datos muy ruidosos

Acciones:
- Aumentar EPOCHS
- Reducir BATCH_SIZE (si es posible)
- Limpiar datos de entrada
- Aumentar LATENT_DIM
```

---

## 📈 Análisis Comparativo Inter-Clases

### Tabla de Comparación
```
               Bigeminy    NSR      Trigeminy   Patrón
Pearson        0.70±0.08  0.82±0.05  0.68±0.09  NSR > Tri > Big
Spectral       0.65±0.10  0.78±0.07  0.62±0.11  NSR > Big > Tri
Energy         0.75±0.09  0.87±0.06  0.71±0.10  NSR > Big > Tri
SNR            20.1±3.2   24.5±2.1   18.5±3.8   NSR > Big > Tri
```

**Interpretación:**
- NSR debería tener métricas más altas (regular y predecible)
- Trigeminy puede ser más baja que Bigeminy (patrón menos evidente)
- Si no sigue este patrón → Revisar datos o modelo

---

## 🎓 Conclusión

Las métricas cuantificadas permiten:

1. **Validación objetiva** de calidad sin subjetividad
2. **Identificación específica** de problemas
3. **Comparación consistente** entre ejecuciones
4. **Toma de decisiones** informada sobre aceptabilidad

Use esta guía para interpretar sus resultados y tomar decisiones sobre si la síntesis es suficientemente buena para su aplicación.

---

**Última actualización:** Diciembre 11, 2025

import os
import io
import zipfile

import streamlit as st
import neurokit2 as nk
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import signal
import pywt

# — 1) Configuración de la página en modo ancho —
st.set_page_config(page_title="ECG Synthetic Signal Generator", layout="wide")

# — 2) Sidebar: logo, ABOUT, CREDITS —
#logo_path = "logo.png"
#if os.path.exists(logo_path):
#    st.sidebar.image(logo_path, width=180)
#else:
#    st.sidebar.info("No se encontró 'logo.png'. Puedes subir tu logo aquí:")
#    uploaded_logo = st.sidebar.file_uploader("Upload logo", type=["png","jpg","jpeg"])
#    if uploaded_logo:
#        st.sidebar.image(uploaded_logo, width=180)

st.sidebar.image("https://raw.githubusercontent.com/MSMRo/EKG_signal_GEN-GUI/refs/heads/main/img/logo.png", width=185)
st.sidebar.markdown("## ABOUT")
st.sidebar.markdown(
    "Esta aplicación genera señales ECG sintéticas "
    "con ruido y baseline wander, y realiza análisis en frecuencia "
    "(FFT, STFT y CWT)."
)
st.sidebar.markdown("---")
st.sidebar.markdown("## CREDITS")
st.sidebar.markdown(
    "- Desarrollo: Ing. Moises Meza Rodriguez\n"
    "- Biblioteca: NeuroKit2, Numpy, PyWavelets\n"
    "- Visualización: Plotly\n"
)
st.sidebar.markdown("---")

# — 3) Frecuencia de muestreo configurable —
sampling_rate = st.sidebar.number_input(
    "Sampling Rate (Hz)",
    min_value=50,
    max_value=2000,
    value=250,
    step=1,
    help="Frecuencia de muestreo para la simulación y los análisis"
)

# — 4) Parámetros de la señal en el sidebar —
signal_type     = st.sidebar.selectbox("ECG Signal Type", ["Normal (regular)"])
window_size     = st.sidebar.slider("Window Size (s)",               1.0, 10.0, 4.0, step=0.1)
isoelectric     = st.sidebar.slider("Isoelectric Wander Amplitude (mV)", 0.0, 1.0, 0.05, step=0.01)
pr_delay        = st.sidebar.slider("P–QRS Delay (ms)",                0,   50,    25)
qt_delay        = st.sidebar.slider("QRS–T Delay (ms)",                0,   50,    25)
bpm             = st.sidebar.slider("BPM (40–140)",                  40,  140,    90)
bpm_noise_amp   = st.sidebar.slider("BPM Noise σ (bpm)",             0.0, 10.0,   1.0, step=0.5)
white_noise_amp = st.sidebar.slider("White Noise (mV)",              0.0,  0.5,   0.04, step=0.01)
pl_noise_amp    = st.sidebar.slider("50 Hz Noise (mV)",              0.0,  1.0,   0.01, step=0.01)
generate        = st.sidebar.button("Generate ECG")

# — 5) Inicializar session_state para la señal —
if "ecg_df" not in st.session_state:
    st.session_state.ecg_df = None

# — 6) Generación de la señal y guardado en session_state —
if generate:
    duration_sec = int(window_size)
    hr_variation = float(bpm + np.random.normal(0, bpm_noise_amp))

    ecg_clean = nk.ecg_simulate(
        duration      = duration_sec,
        sampling_rate = sampling_rate,
        heart_rate    = hr_variation,
        method        = "ecgsyn",
        noise         = 0,
    )

    t = np.arange(len(ecg_clean)) / sampling_rate
    baseline_wander = isoelectric * np.sin(2 * np.pi * 0.33 * t)
    noise_white     = np.random.normal(0, white_noise_amp, size=len(t))
    noise_pl        = pl_noise_amp * np.sin(2 * np.pi * 50 * t)
    ecg_noisy       = ecg_clean + baseline_wander + noise_white + noise_pl

    st.session_state.ecg_df = pd.DataFrame({
        "Time (s)":     t,
        "Voltage (mV)": ecg_noisy
    })

# — 7) Título en la parte principal —
st.title("ECG Synthetic Signal Generator")

# — 8) Mostrar gráficos y ZIP de descarga si hay datos generados —
if st.session_state.ecg_df is not None:
    df = st.session_state.ecg_df
    t  = df["Time (s)"].values
    x  = df["Voltage (mV)"].values

    # — 8.1) Plot ECG —
    fig_ecg = go.Figure()
    fig_ecg.add_trace(go.Scatter(x=t, y=x, mode="lines", line=dict(width=2)))
    fig_ecg.update_layout(
        title="ECG Synthetic Signal",
        xaxis_title="Time (s)",
        yaxis_title="Voltage (mV)",
        margin=dict(l=40, r=40, t=40, b=40)
    )
    st.plotly_chart(fig_ecg, use_container_width=True)
    # — Botón de descarga CSV ———————————————————————————————
    #df = pd.DataFrame({
    #    "Time (s)":      t,
    #    "Voltage (mV)":  ecg_noisy
    #})
    csv2 = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download data as CSV",
        data=csv2,
        file_name="ecg_data.csv",
        mime="text/csv",
    )

    # — 8.2) Plot FFT —
    n = len(x)
    freqs   = np.fft.rfftfreq(n, d=1/sampling_rate)
    fft_vals= np.abs(np.fft.rfft(x)) / n
    fig_fft = go.Figure()
    fig_fft.add_trace(go.Scatter(x=freqs, y=fft_vals, mode="lines", line=dict(width=2)))
    fig_fft.update_layout(
        title="FFT (Magnitude Spectrum)",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Amplitude",
        margin=dict(l=40, r=40, t=40, b=40)
    )
    st.plotly_chart(fig_fft, use_container_width=True)

    # — 8.3) Plot STFT —
    f_stft, t_stft, Zxx = signal.stft(x, sampling_rate, nperseg=128)
    fig_stft = go.Figure(data=go.Heatmap(
        z=np.abs(Zxx),
        x=t_stft,
        y=f_stft,
        colorscale="Viridis"
    ))
    fig_stft.update_layout(
        title="STFT (Magnitude)",
        xaxis_title="Time (s)",
        yaxis_title="Frequency (Hz)",
        margin=dict(l=40, r=40, t=40, b=40)
    )
    st.plotly_chart(fig_stft, use_container_width=True)

    # — 8.4) Plot CWT/Morlet —
    scales        = np.arange(1, 128)
    coeffs, freqs_cwt = pywt.cwt(x, scales, "morl", sampling_period=1/sampling_rate)
    fig_cwt = go.Figure(data=go.Heatmap(
        z=np.abs(coeffs),
        x=t,
        y=freqs_cwt,
        colorscale="Viridis"
    ))
    fig_cwt.update_layout(
        title="CWT (Morlet)",
        xaxis_title="Time (s)",
        yaxis_title="Frequency (Hz)",
        margin=dict(l=40, r=40, t=40, b=40)
    )
    st.plotly_chart(fig_cwt, use_container_width=True)

    # — 9) Empaquetar todo en un ZIP y ofrecer descarga —
    #zip_buffer = io.BytesIO()
    #with zipfile.ZipFile(zip_buffer, mode="w") as zf:
    #    # Señal CSV
    #    csv_bytes = df.to_csv(index=False).encode("utf-8")
    #    zf.writestr("ecg_data.csv", csv_bytes)
    #    # Imágenes PNG
    #    zf.writestr("ecg_plot.png",  fig_ecg.to_image(format="png"))
    #    zf.writestr("fft_plot.png",  fig_fft.to_image(format="png"))
    #    zf.writestr("stft_plot.png", fig_stft.to_image(format="png"))
    #zip_buffer.seek(0)

    #st.markdown("### Download all results")
    #st.download_button(
    #    label="Download ECG data + plots (ZIP)",
    #    data=zip_buffer,
    #    file_name="ecg_results.zip",
    #    mime="application/zip"
    #)

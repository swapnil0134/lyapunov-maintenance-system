import streamlit as st
import numpy as np
import time
import plotly.graph_objects as go
from datetime import datetime
import collections
import sys
import os

# Add current directory to path so we can import modules
sys.path.append(os.getcwd())

try:
    from chaos_industrial.detection.industrial_detector import IndustrialFailureDetector
except ImportError:
    st.error("Could not import chaos_industrial module. Please make sure you are in the root of the repository.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="ChaosGuard Industrial",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to match the screenshot look
st.markdown("""
<style>
    .stApp {
        background: #0e1117;
    }
    .metric-card {
        background-color: #1f2937;
        border-radius: 0.5rem;
        padding: 1rem;
        border: 1px solid #374151;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: bold;
        color: #60a5fa;
    }
    .metric-label {
        color: #9ca3af;
        font-size: 0.875rem;
    }
    .stAlert {
        background-color: #1f2937;
        color: #e5e7eb;
        border: 1px solid #374151;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'detector' not in st.session_state:
    st.session_state.detector = IndustrialFailureDetector(window_size=1000, sample_rate=1000)
    st.session_state.data_buffer = collections.deque(maxlen=1000)
    st.session_state.events = []
    st.session_state.running = False
    st.session_state.sample_count = 0

def generate_data(mode, t):
    """Generate simulated sensor data based on selected mode"""
    base_freq = 50  # Hz
    noise = np.random.normal(0, 0.1)

    if mode == "Normal Operation":
        x = np.sin(2 * np.pi * base_freq * t) + noise
        y = np.cos(2 * np.pi * base_freq * t) + noise
        z = 9.8 + noise

    elif mode == "Inject Unbalance":
        x = 2 * np.sin(2 * np.pi * base_freq * t) + noise
        y = np.cos(2 * np.pi * base_freq * t) + noise
        z = 9.8 + noise

    elif mode == "Inject Misalignment":
        # 2X harmonic
        x = np.sin(2 * np.pi * base_freq * t) + 0.5 * np.sin(4 * np.pi * base_freq * t) + noise
        y = np.cos(2 * np.pi * base_freq * t) + 0.5 * np.cos(4 * np.pi * base_freq * t) + noise
        z = 9.8 + noise

    elif mode == "Simulate Bearing Wear":
        # High frequency
        x = np.sin(2 * np.pi * base_freq * t) + 0.5 * np.sin(2 * np.pi * 800 * t) + noise
        y = np.cos(2 * np.pi * base_freq * t) + 0.5 * np.sin(2 * np.pi * 800 * t) + noise
        z = 9.8 + noise

    elif mode == "Mechanical Looseness":
        # Multiple harmonics
        x = np.sin(2 * np.pi * base_freq * t) + 0.3 * np.sin(4 * np.pi * base_freq * t) + \
            0.2 * np.sin(6 * np.pi * base_freq * t) + noise
        y = np.cos(2 * np.pi * base_freq * t) + noise
        z = 9.8 + noise

    else:
        x, y, z = 0, 0, 9.8

    return x, y, z

# Sidebar
with st.sidebar:
    st.title("🤖 ChaosGuard Industrial")
    st.caption("V.2.5.0-ALPHA // NONLINEAR DYNAMICS ENGINE")

    st.subheader("FAULT INJECTION CONTROL")
    mode = st.radio(
        "Select Operation Mode",
        ["Normal Operation", "Inject Unbalance", "Inject Misalignment", "Simulate Bearing Wear", "Mechanical Looseness"],
        key="mode"
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start Monitoring", type="primary"):
            st.session_state.running = True
    with col2:
        if st.button("Stop"):
            st.session_state.running = False

    st.divider()

    st.subheader("System Events")
    event_container = st.empty()

    # Display recent events
    events_html = ""
    for event in list(st.session_state.events)[-10:]:
        color = "#ef4444" if "ALERT" in event else "#3b82f6"
        events_html += f"<div style='color: {color}; font-size: 0.8em; margin-bottom: 4px;'>{event}</div>"
    event_container.markdown(events_html, unsafe_allow_html=True)

# Main Layout
col_top1, col_top2 = st.columns([2, 1])

with col_top1:
    st.subheader("SYSTEM CONDITION")
    status_placeholder = st.empty()

with col_top2:
    st.subheader("GLOBAL ANOMALY SCORE")
    score_placeholder = st.empty()

# Metrics Row
col_m1, col_m2, col_m3 = st.columns(3)
with col_m1:
    lyapunov_metric = st.empty()
with col_m2:
    corr_dim_metric = st.empty()
with col_m3:
    entropy_metric = st.empty()

# Charts
st.subheader("Time Domain Signal (Vibration)")
time_chart = st.empty()

col_bottom1, col_bottom2 = st.columns(2)
with col_bottom1:
    st.subheader("Frequency Domain (FFT)")
    fft_chart = st.empty()
with col_bottom2:
    st.subheader("Phase Space Attractor (Reconstruction)")
    phase_chart = st.empty()

def update_ui():
    status = st.session_state.detector.get_current_status()

    # Status
    status_text = status['status']
    status_color = "#22c55e" if "NORMAL" in status_text else ("#ef4444" if "CRITICAL" in status_text else "#eab308")
    status_placeholder.markdown(
        f"""<div style='background-color: #111827; padding: 20px; border-radius: 10px; border-left: 5px solid {status_color}'>
            <h1 style='color: {status_color}; margin: 0;'>{status_text.replace('✓ ', '').replace('⚠️ ', '').replace('🚨 ', '')}</h1>
            <p style='color: #6b7280; margin: 0;'>Active Mode: {mode}</p>
        </div>""",
        unsafe_allow_html=True
    )

    # Score
    score = status['anomaly_score']
    score_placeholder.markdown(
        f"""<div style='background-color: #111827; padding: 20px; border-radius: 10px; text-align: right;'>
            <h1 style='color: {status_color}; margin: 0; font-size: 3em;'>{score:.2f}</h1>
        </div>""",
        unsafe_allow_html=True
    )

    # Metrics
    if st.session_state.detector.history['chaos_metrics']:
        latest_chaos = st.session_state.detector.history['chaos_metrics'][-1]

        # Helper for metric card
        def metric_card(label, value, desc):
            return f"""
            <div class='metric-card'>
                <div class='metric-label'>{label}</div>
                <div class='metric-value'>{value:.4f}</div>
                <div style='font-size: 0.7em; color: #6b7280;'>{desc}</div>
            </div>
            """

        lyapunov_metric.markdown(metric_card("LYAPUNOV EXPONENT (Λ₁)", latest_chaos.get('lyapunov', 0), "Measures chaotic divergence. >0.15 indicates critical instability."), unsafe_allow_html=True)
        corr_dim_metric.markdown(metric_card("CORRELATION DIM (D₂)", latest_chaos.get('correlation_dimension', 0), "Fractal complexity. High values suggest mechanical looseness."), unsafe_allow_html=True)
        entropy_metric.markdown(metric_card("APPROX. ENTROPY (ApEn)", latest_chaos.get('approximate_entropy', 0), "Signal regularity. Low values indicate strong unbalance (periodic)."), unsafe_allow_html=True)

    # Charts logic
    if len(st.session_state.data_buffer) > 100:
        data = np.array(st.session_state.data_buffer)

        # Time Domain
        fig_time = go.Figure()
        fig_time.add_trace(go.Scatter(y=data[-500:], mode='lines', line=dict(color='#3b82f6', width=1)))
        fig_time.update_layout(
            template="plotly_dark",
            margin=dict(l=0, r=0, t=0, b=0),
            height=250,
            xaxis=dict(showgrid=True, gridcolor='#374151'),
            yaxis=dict(showgrid=True, gridcolor='#374151')
        )
        time_chart.plotly_chart(fig_time, use_container_width=True)

        # Frequency Domain
        # We need to perform FFT on the data buffer
        freqs = np.fft.rfftfreq(len(data), 1/1000)
        fft_vals = np.abs(np.fft.rfft(data))

        # Filter for display (0-1000 Hz)
        mask = freqs < 1000

        fig_fft = go.Figure()
        fig_fft.add_trace(go.Bar(x=freqs[mask], y=fft_vals[mask], marker_color='#8b5cf6'))
        fig_fft.update_layout(
            template="plotly_dark",
            margin=dict(l=0, r=0, t=0, b=0),
            height=250,
            xaxis_title="Frequency (Hz)"
        )
        fft_chart.plotly_chart(fig_fft, use_container_width=True)

        # Phase Space
        # Simple delay embedding (tau=10)
        tau = 10
        if len(data) > tau:
            x_phase = data[:-tau]
            y_phase = data[tau:]

            fig_phase = go.Figure()
            fig_phase.add_trace(go.Scatter(
                x=x_phase[-1000:], y=y_phase[-1000:],
                mode='markers',
                marker=dict(size=2, color='#3b82f6', opacity=0.5)
            ))
            fig_phase.update_layout(
                template="plotly_dark",
                margin=dict(l=0, r=0, t=0, b=0),
                height=250,
                showlegend=False
            )
            phase_chart.plotly_chart(fig_phase, use_container_width=True)

# Main Loop
if st.session_state.running:
    # Process a batch of samples to speed up
    batch_size = 50
    current_t = st.session_state.sample_count / 1000.0

    for _ in range(batch_size):
        current_t += 0.001
        x, y, z = generate_data(mode, current_t)

        # Update detector
        st.session_state.detector.add_accelerometer_sample(x, y, z)

        # Update buffer for plotting (using magnitude)
        mag = np.sqrt(x**2 + y**2 + z**2)
        st.session_state.data_buffer.append(mag)

    st.session_state.sample_count += batch_size

    # Add event logs if status changed or periodically
    status = st.session_state.detector.get_current_status()
    if status['anomaly_score'] > 0.5 and (not st.session_state.events or "ALERT" not in st.session_state.events[-1]):
        timestamp = datetime.now().strftime("%H:%M:%S")
        st.session_state.events.append(f"[{timestamp}] ALERT: High anomaly score detected: {status['anomaly_score']:.2f}")

    # Update UI
    update_ui()

    # Rerun
    time.sleep(0.01)
    st.rerun()

else:
    # Initial static view
    update_ui()
    st.info("Click 'Start Monitoring' to begin real-time simulation.")

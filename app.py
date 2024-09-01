import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from scipy.signal import stft

# Function to load the trained model
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()

# Title and description
st.title("Micro-Doppler Target Classification")
st.markdown("""
### 🛸 Identify if the Signal is from a Drone or a Bird
Upload a signal file to classify it based on Micro-Doppler signatures.
""")

# File uploader
uploaded_file = st.file_uploader("Upload Signal File (CSV)", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded file
    data = pd.read_csv(uploaded_file, header=None)
    signal = data.values.flatten()

    st.subheader("Uploaded Signal (Time-Domain)")
    st.line_chart(signal)

    # Perform STFT
    f, t, Zxx = stft(signal, fs=1000, nperseg=256)
    spectrogram = np.abs(Zxx)

    # Display Spectrogram
    st.subheader("Spectrogram (Frequency-Time Domain)")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.pcolormesh(t, f, spectrogram, shading='gouraud')
    ax.set_ylabel('Frequency [Hz]')
    ax.set_xlabel('Time [sec]')
    st.pyplot(fig)

    # Feature Extraction
    mean_freq = np.mean(spectrogram, axis=0)
    max_freq = np.max(spectrogram, axis=0)
    std_freq = np.std(spectrogram, axis=0)
    features = np.hstack([mean_freq, max_freq, std_freq])
    features = features.reshape(1, -1)

    # Ensure features match model's expected input size
    expected_feature_size = model.n_features_in_
    if features.shape[1] > expected_feature_size:
        features = features[:, :expected_feature_size]
    elif features.shape[1] < expected_feature_size:
        features = np.pad(features, ((0,0),(0, expected_feature_size - features.shape[1])), 'constant')

    # Prediction
    prediction = model.predict(features)[0]
    prediction_proba = model.predict_proba(features)[0]

    # Display Classification Result
    st.subheader("Classification Result")
    target = "Drone" if prediction == 1 else "Bird"
    confidence = np.max(prediction_proba) * 100
    st.markdown(f"**Result:** {target}")
    st.markdown(f"**Confidence:** {confidence:.2f}%")
    st.progress(confidence / 100)
else:
    st.info("Please upload a CSV file containing the signal data.")

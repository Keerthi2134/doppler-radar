import numpy as np
import pandas as pd

# Constants
np.random.seed(42)
duration = 5  # seconds
sampling_rate = 1000  # Hz
time = np.linspace(0, duration, duration * sampling_rate)

# Helper function to generate micro-Doppler signature
def generate_signal(frequency, amplitude, noise_level=0.1):
    signal = amplitude * np.sin(2 * np.pi * frequency * time)
    noise = noise_level * np.random.normal(size=time.shape)
    return signal + noise

# Generate drone signals (e.g., propeller rotation)
def generate_drone_signal():
    base_frequency = 50  # Hz for the body movement
    modulated_frequency = 200  # Hz for propeller blades
    signal_body = generate_signal(base_frequency, amplitude=1.0, noise_level=0.2)
    signal_propeller = generate_signal(modulated_frequency, amplitude=0.5, noise_level=0.2)
    return signal_body + signal_propeller

# Generate bird signals (e.g., wing beats)
def generate_bird_signal():
    wing_beat_frequency = 10  # Hz for wing beats
    signal_body = generate_signal(wing_beat_frequency, amplitude=1.0, noise_level=0.3)
    return signal_body

# Generate dataset
def generate_dataset(num_samples=100):
    data = []
    labels = []
    for _ in range(num_samples // 2):
        # Generate drone data
        drone_signal = generate_drone_signal()
        data.append(drone_signal)
        labels.append(1)  # 1 for drone
        
        # Generate bird data
        bird_signal = generate_bird_signal()
        data.append(bird_signal)
        labels.append(0)  # 0 for bird
        
    return np.array(data), np.array(labels)

# Generate and save the dataset
data, labels = generate_dataset(200)
df = pd.DataFrame(data)
df['label'] = labels
df.to_csv('synthetic_micro_doppler_dataset.csv', index=False)
print("Synthetic dataset generated and saved as 'synthetic_micro_doppler_dataset.csv'.")

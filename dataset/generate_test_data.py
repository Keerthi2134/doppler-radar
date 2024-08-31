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

# Generate random test data
def generate_random_test_data(num_samples=10):
    data = []
    labels = []
    for _ in range(num_samples):
        # Randomly choose between generating a drone or bird signal
        if np.random.choice([0, 1]) == 1:
            signal = generate_drone_signal()
            label = 1  # Drone
        else:
            signal = generate_bird_signal()
            label = 0  # Bird
        
        data.append(signal)
        labels.append(label)
    
    return np.array(data), np.array(labels)

# Generate and save the test data
def save_test_data(filename='test_micro_doppler_signals.csv', num_samples=10):
    data, labels = generate_random_test_data(num_samples)
    df = pd.DataFrame(data)
    df['label'] = labels
    df.to_csv(filename, index=False, header=False)  # Save without header
    print(f"Test data generated and saved as '{filename}'.")

# Example usage
if __name__ == '__main__':
    save_test_data()

import numpy as np
from scipy.signal import stft

# Example: Load your radar data (time-series) from a file
# Assuming `data` is a 1D NumPy array representing the radar signal

def load_data(file_path):
    data = np.load(file_path)  # Replace with your actual data loading method
    return data

# Perform Short-Time Fourier Transform (STFT) for time-frequency analysis
def perform_stft(data, fs=1000, nperseg=256):
    f, t, Zxx = stft(data, fs=fs, nperseg=nperseg)
    return f, t, np.abs(Zxx)

# Example usage:
file_path = 'path_to_your_data_file.npy'
data = load_data(file_path)
frequencies, times, spectrogram = perform_stft(data)

# Visualize the spectrogram
import matplotlib.pyplot as plt

plt.pcolormesh(times, frequencies, spectrogram)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.title('STFT Spectrogram')
plt.colorbar()
plt.show()
# Feature extraction from the spectrogram
def extract_features(spectrogram):
    mean_frequency = np.mean(spectrogram, axis=0)  # Mean frequency over time
    energy = np.sum(spectrogram**2, axis=0)  # Energy of the signal over time
    return mean_frequency, energy

# Example usage:
mean_freq, energy = extract_features(spectrogram)
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Example: Load your features and labels
# Assuming X is your feature matrix and y are your labels (0 for birds, 1 for drones)

X = np.array([mean_freq, energy]).T  # Feature matrix
y = np.array([0, 1])  # Labels (replace with your actual labels)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train an SVM classifier
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

# Evaluate the model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

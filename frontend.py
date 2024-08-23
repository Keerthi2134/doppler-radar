import tkinter as tk
from tkinter import filedialog
import numpy as np

def classify_signal():
    # Load the data file
    file_path = filedialog.askopenfilename()
    data = load_data(file_path)
    
    # Perform STFT and extract features
    _, _, spectrogram = perform_stft(data)
    mean_freq, energy = extract_features(spectrogram)
    
    # Classify using the loaded model
    features = np.array([mean_freq, energy]).reshape(1, -1)
    prediction = model.predict(features)
    
    result = 'Drone' if prediction[0] == 1 else 'Bird'
    result_label.config(text=f'Class: {result}')

# GUI setup
root = tk.Tk()
root.title("Micro-Doppler Classification")

classify_button = tk.Button(root, text="Classify Signal", command=classify_signal)
classify_button.pack()

result_label = tk.Label(root, text="Class: ")
result_label.pack()

root.mainloop()

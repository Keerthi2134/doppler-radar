import tkinter as tk
from tkinter import filedialog
import numpy as np
import pandas as pd
import pickle

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load data function
def load_data(file_path):
    df = pd.read_csv(file_path, header=None)
    data = df.iloc[:, :-1].values.flatten()
    return data

# GUI function to classify signal
def classify_signal():
    file_path = filedialog.askopenfilename()
    data = load_data(file_path)
    
    features = data.reshape(1, -1)
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

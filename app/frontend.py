import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import pandas as pd
import pickle
from tkinter import ttk

# Load the trained model
model_path = 'model.pkl'
try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    messagebox.showerror("Error", f"Model file {model_path} not found.")
    exit()
except pickle.PickleError:
    messagebox.showerror("Error", "Error loading the model.")
    exit()

# Load data function
def load_data(file_path):
    df = pd.read_csv(file_path, header=None)
    data = df.iloc[0, :5000].values  # Ensure only 5000 features are selected
    return data

# GUI function to classify signal
def classify_signal():
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if not file_path:
        return
    
    try:
        data = load_data(file_path)
        
        # Ensure the input data has exactly 5000 features
        if len(data) != 5000:
            messagebox.showerror("Error", "Incorrect data format. Expected 5000 features.")
            return
        
        features = data.reshape(1, -1)
        progress_label.config(text="Classifying...")
        root.update()  # Update the UI
        
        prediction = model.predict(features)
        result = 'Drone' if prediction[0] == 1 else 'Bird'
        result_label.config(text=f'Class: {result}', foreground="green")
        progress_label.config(text="Classification Complete")
        
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

# GUI Setup
root = tk.Tk()
root.title("Micro-Doppler Classification")
root.geometry("500x350")
root.configure(bg='#f0f0f0')

# Title Label
title_label = tk.Label(root, text="Micro-Doppler Classification", font=("Helvetica", 18, "bold"), bg='#f0f0f0', fg='#333')
title_label.pack(pady=20)

# Frame for buttons and result
frame = tk.Frame(root, bg='#f0f0f0')
frame.pack(pady=10)

# Classify Button with styling
classify_button = ttk.Button(frame, text="Choose Signal File", command=classify_signal, style="TButton")
classify_button.grid(row=0, column=0, padx=20, pady=10)

# Result Label
result_label = tk.Label(frame, text="Class: ", font=("Helvetica", 16), bg='#f0f0f0', fg='#555')
result_label.grid(row=1, column=0, padx=20, pady=10)

# Progress Label
progress_label = tk.Label(frame, text="", font=("Helvetica", 12), bg='#f0f0f0', fg='#888')
progress_label.grid(row=2, column=0, padx=20, pady=10)

# Styling for buttons (using ttk style)
style = ttk.Style()
style.configure("TButton", font=("Helvetica", 14), padding=8, background="#007ACC", foreground="black")
style.map("TButton", background=[('active', '#005F9E')], foreground=[('active', 'white')])

# Footer Label
footer_label = tk.Label(root, text="Developed for Hackathon 2024", font=("Helvetica", 12), bg='#f0f0f0', fg='#888')
footer_label.pack(side=tk.BOTTOM, pady=10)

root.mainloop()

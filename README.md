# Smart India Hackathon 2024: Micro-Doppler Based Target Classification

**Team Name**: Whiskey Tango Foxtrot.


## Problem Statement Overview

**Problem Statement ID**: SIH1606  
**Theme**: Robotics and Drones  
**Category**: Software/Hardware

Micro-Doppler-based target classification is crucial in distinguishing drones from birds in radar surveillance systems. The increasing prevalence of drones in sensitive airspace requires a robust and accurate system to differentiate between these targets to ensure safety and security.

## Proposed Solution

Our proposed solution involves leveraging micro-Doppler signatures obtained from radar data to classify drones and birds effectively. The solution employs advanced signal processing techniques and machine learning algorithms to extract and select meaningful features for high-accuracy classification. This project integrates both software and hardware components for real-time deployment and scalability.

## Technical Overview

### 1. **Data Acquisition and Preprocessing**

- **Radar Data Collection**: Utilizes simulated radar data to generate micro-Doppler signatures for various targets such as drones and birds.
- **Preprocessing Techniques**: 
  - **Noise Reduction**: Applies noise filtering to clean the radar data.
  - **Time-Frequency Transformations**: Utilizes Short-Time Fourier Transform (STFT) and Continuous Wavelet Transform (CWT) to enhance signal clarity and provide detailed time-frequency representations of micro-Doppler signatures.

### 2. **Feature Extraction and Selection**

- **Feature Extraction**:
  - Extracts time-frequency patterns, statistical measures, and target-specific features related to shape, motion, and temporal characteristics from the radar data.
- **Feature Selection**:
  - Employs techniques such as Principal Component Analysis (PCA) and Linear Discriminant Analysis (LDA) to select the most relevant features, reducing dimensionality and optimizing model performance.

### 3. **Classification and Model Training**

- **Machine Learning Models**:
  - **Support Vector Machines (SVMs)**, **k-Nearest Neighbors (k-NN)**, **Convolutional Neural Networks (CNNs)**, and **Long Short-Term Memory (LSTM) networks** are explored for target classification.
- **Training and Evaluation**:
  - The models are trained using labeled data and evaluated with cross-validation techniques and performance metrics such as accuracy and recall.

### 4. **Deployment**

- **Real-Time Application**:
  - Implements the classification software for real-time applications, optimized for low-latency processing, crucial for security and surveillance scenarios.

## Repository Structure

The repository is organized into three main folders: `app`, `dataset`, and `model`.

### `app/`

Contains the backend and frontend components for the application.

- **`backend.py`**:  
  A Flask-based REST API to classify radar data. It loads the pre-trained SVM model from `model.pkl` and processes incoming JSON requests to predict whether the target is a drone or a bird.
  - **Endpoint**: `/classify` (POST)
  - **Input**: JSON object containing radar data features.
  - **Output**: JSON response with classification result (`drone` or `bird`).

- **`frontend.py`**:  
  A Tkinter-based GUI application for local use, allowing users to classify micro-Doppler signals from radar data files.
  - **Functionality**:
    - Provides a file dialog for users to upload radar data files.
    - Displays the classification result (`Drone` or `Bird`) in the GUI.

### `dataset/`

Contains the script to generate synthetic radar data for training and testing purposes.

- **`generate_dataset.py`**:  
  Generates synthetic micro-Doppler signatures for drones and birds using sine wave functions modulated by different frequencies. The generated dataset is saved as `synthetic_micro_doppler_dataset.csv`.
  - **Data Characteristics**:
    - Drones: Modeled with a combination of base and modulated frequencies (e.g., 50 Hz and 200 Hz).
    - Birds: Modeled with wing-beat frequencies (e.g., 10 Hz).
  - **Output**: CSV file containing synthetic radar signals and corresponding labels.

### `model/`

Contains the script to train and save the machine learning model.

- **`train_model.py`**:  
  Trains an SVM classifier on the generated dataset and evaluates its performance.
  - **Process**:
    - Loads and preprocesses the dataset.
    - Splits the data into training and testing sets.
    - Trains an SVM model with a linear kernel.
    - Evaluates model performance using accuracy as the metric.
    - Saves the trained model as `model.pkl` for use in the backend.
  - **Output**: Serialized model file `model.pkl`.

## Technical Stack

- **Programming Languages**:
  - Python (for data processing, feature extraction, and machine learning)
- **Frameworks and Libraries**:
  - **Flask**: For building the backend REST API.
  - **Tkinter**: For building the GUI frontend.
  - **NumPy**, **Pandas**, **SciPy**: For data handling and numerical computations.
  - **Scikit-Learn**: For machine learning model training and evaluation.
- **Signal Processing Tools**:
  - **STFT** and **CWT**: For time-frequency analysis.
- **Machine Learning Models**:
  - **SVM**: Used as the primary model for classification.

## Benefits and Advantages

1. **Improved Target Differentiation**: Accurately distinguishes between drones and birds using micro-Doppler signatures.
2. **Noise Reduction and Clutter Mitigation**: Preprocessing steps reduce noise and enhance signal clarity, improving classification reliability.
3. **Real-Time Processing**: Designed for low-latency deployment in real-world applications, ensuring timely decision-making.
4. **Scalability**: The solution is adaptable to various environmental conditions and scalable for larger datasets.
5. **Enhanced Security**: Facilitates better surveillance capabilities for security agencies and operators.

## How to Use

1. **Clone the Repository**:  
   `git clone <repository_url>`

2. **Install Dependencies**:  
   Ensure Python is installed and run `pip install -r requirements.txt` to install the necessary libraries.

3. **Generate Dataset**:
   - Run `generate_dataset.py` from the `dataset/` folder to create the synthetic radar dataset.

4. **Train the Model**:
   - Run `train_model.py` from the `model/` folder to train the SVM model and save it as `model.pkl`.

5. **Run the Backend API**:
   - Navigate to the `app/` folder and run `backend.py` to start the Flask server.

6. **Run the Frontend Application**:
   - Execute `frontend.py` to launch the GUI for radar data classification.

## Project Scope 

The project focuses on using micro-Doppler radar signatures to classify aerial objects, distinguishing drones from birds in real-time surveillance applications. This solution enhances security, airspace management, and wildlife monitoring.

**Real-Life Applications**

-Aviation Security: Preventing drone intrusions near airports.
-Military and Defense: Detecting unauthorized drones in restricted areas.
-Wildlife Conservation: Monitoring bird migration patterns.
-Industrial Facilities: Securing critical infrastructure from drone threats.

**Project Scope Boundaries**

**Included:**

-Radar data acquisition and micro-Doppler feature extraction.
-Machine learning-based classification models (SVM, CNN, LSTM).
-Preprocessing techniques for noise reduction and signal enhancement.
-Real-time processing through a backend API and GUI.

**Excluded:**

-Hardware implementation of radar sensors (uses simulated data).
-Multi-class classification beyond drones and birds.
-Environmental factors like weather interference in classification.

**Deliverables:**

-A trained machine learning model for drone-bird classification.
-Backend API to process radar signals and return classification results.
-GUI-based frontend for user-friendly interaction.
-Synthetic dataset generation script for training and evaluation.
-Achieve a classification accuracy of 90%+ on test datasets.
-Process radar signals in real-time (low-latency inference).
-Successful deployment in security and surveillance applications.

## Screenshots
<img width="946" alt="image" src="https://github.com/user-attachments/assets/9bf25817-0436-4190-a15d-74552aca4f14">
<img width="946" alt="image" src="https://github.com/user-attachments/assets/00995a90-e0a6-452b-bc89-e1a9ecb3a9fd">
<img width="947" alt="image" src="https://github.com/user-attachments/assets/b5339a23-6719-4101-af00-5cf0106fa0f1">
<img width="944" alt="image" src="https://github.com/user-attachments/assets/bae5478e-ce6b-4297-a786-c2bbf570ed37">



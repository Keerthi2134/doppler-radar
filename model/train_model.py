import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle

# Load the dataset
df = pd.read_csv('synthetic_micro_doppler_dataset.csv')
X = df.iloc[:, :-1].values  # All columns except the last
y = df['label'].values       # Last column as labels

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train an SVM classifier
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

# Evaluate the model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Save the trained model
with open('model.pkl', 'wb') as f:
    pickle.dump(clf, f)
print("Model saved as 'model.pkl'.")

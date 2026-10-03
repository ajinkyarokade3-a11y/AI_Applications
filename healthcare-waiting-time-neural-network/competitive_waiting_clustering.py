# ==================================================
# PART B: UNSUPERVISED LEARNING (COMPETITIVE LEARNING)
# ==================================================

import numpy as np
import pandas as pd

# Healthcare Waiting-Time Dataset (Without Target)
data_unsupervised = {
    "Patients_Waiting":    [15, 22, 28, 35, 60, 68, 75, 85],
    "Available_Doctors":   [8, 7, 6, 5, 3, 3, 2, 2],
    "Appointment_Count":   [25, 30, 40, 45, 70, 78, 85, 95],
    "Emergency_Cases":     [2, 3, 4, 5, 9, 10, 12, 14],
    "Avg_Consultation_Time": [12, 15, 18, 20, 32, 36, 40, 45]
}
df_unsup = pd.DataFrame(data_unsupervised)
print("Unsupervised Input Dataset:")
print(df_unsup)

# Normalize Data
X_unsup = df_unsup.values / max_vals

# Initialize Two Competitive Neurons (5 features each)
weights_unsup = np.array([
    [0.2, 0.5, 0.3, 0.2, 0.3],
    [0.7, 0.2, 0.8, 0.6, 0.6]
])
learning_rate_unsup = 0.3
epochs_unsup = 10

# Competitive Learning Loop
for epoch in range(epochs_unsup):
    for x in X_unsup:
        distances = np.linalg.norm(weights_unsup - x, axis=1)
        winner = np.argmin(distances)
        weights_unsup[winner] = weights_unsup[winner] + learning_rate_unsup * (x - weights_unsup[winner])

# Assign Clusters
clusters = []
for x in X_unsup:
    distances = np.linalg.norm(weights_unsup - x, axis=1)
    winner = np.argmin(distances)
    clusters.append(winner + 1)

# Display Results
df_unsup["Cluster"] = clusters
print("\n======================================")
print("     UNSUPERVISED LEARNING")
print("======================================")
print(df_unsup)
print("\nFinal Neuron Weights:")
print(weights_unsup)
print("======================================")

# ==================================================
# PART A: SUPERVISED LEARNING (PERCEPTRON LEARNING RULE)
# ==================================================

import numpy as np
import pandas as pd

# Healthcare Waiting-Time Dataset
data_supervised = {
    "Patients_Waiting":    [15, 22, 28, 35, 60, 68, 75, 85],
    "Available_Doctors":   [8, 7, 6, 5, 3, 3, 2, 2],
    "Appointment_Count":   [25, 30, 40, 45, 70, 78, 85, 95],
    "Emergency_Cases":     [2, 3, 4, 5, 9, 10, 12, 14],
    "Avg_Consultation_Time": [12, 15, 18, 20, 32, 36, 40, 45],
    "Target":              [1, 1, 1, 1, 0, 0, 0, 0]
}
df_sup = pd.DataFrame(data_supervised)
print("Supervised Training Dataset:")
print(df_sup)

# Input and Target Extraction
X_sup = df_sup[["Patients_Waiting", "Available_Doctors", "Appointment_Count", "Emergency_Cases", "Avg_Consultation_Time"]].values
y_sup = df_sup["Target"].values

# Normalize Input
max_vals = np.array([100.0, 20.0, 100.0, 20.0, 60.0])
X_sup = X_sup / max_vals

# Initialize Parameters
weights_sup = np.zeros(X_sup.shape[1])
bias_sup = 0.0
learning_rate_sup = 0.1
epochs_sup = 20

# Step Activation Function
def activation(value):
    if value >= 0:
        return 1
    else:
        return 0

# Training Loop
for epoch in range(epochs_sup):
    total_error = 0
    for i in range(len(X_sup)):
        net = np.dot(X_sup[i], weights_sup) + bias_sup
        prediction = activation(net)
        error = y_sup[i] - prediction
        weights_sup = weights_sup + learning_rate_sup * error * X_sup[i]
        bias_sup = bias_sup + learning_rate_sup * error
        total_error += abs(error)
    print("Epoch:", epoch + 1, "Total Error:", total_error)
    if total_error == 0:
        break

print("\nTrained Weights:", weights_sup)
print("Trained Bias:", bias_sup)

# Test New Operational State
patients = float(input("\nEnter number of patients waiting: "))
doctors = float(input("Enter available doctors: "))
appointments = float(input("Enter appointment count: "))
emergencies = float(input("Enter emergency cases: "))
consultation_time = float(input("Enter average consultation time (mins): "))

new_state = np.array([patients, doctors, appointments, emergencies, consultation_time]) / max_vals

net = np.dot(new_state, weights_sup) + bias_sup
prediction = activation(net)

print("\n====================================")
print("     PERCEPTRON CLASSIFICATION")
print("====================================")
print("Patients Waiting       :", patients)
print("Available Doctors      :", doctors)
print("Appointment Count      :", appointments)
print("Emergency Cases        :", emergencies)
print("Avg Consultation Time  :", consultation_time)
if prediction == 1:
    print("Prediction : ACCEPTABLE WAITING")
else:
    print("Prediction : EXCESSIVE WAITING")
print("====================================")

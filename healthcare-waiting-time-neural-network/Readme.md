# Healthcare Waiting-Time Management System Using Neural Networks

This repository contains a Python implementation of basic **supervised** and **unsupervised** neural network learning rules applied to a real-world healthcare operational scenario. The system analyzes hospital operational metrics to classify waiting-time conditions and discover natural operational patterns.

---

## 📋 Overview

In healthcare operations, managing patient queues and resource allocation efficiently is critical. This project demonstrates foundational neural network concepts using two distinct learning approaches:
1. **Supervised Learning (Perceptron Learning Rule):** Classifies hospital operational states into binary categories: **Acceptable Waiting** or **Excessive Waiting**.
2. **Unsupervised Learning (Competitive Learning Rule):** Automatically clusters and discovers hidden operational patterns from hospital workload data without relying on predefined target labels.

---

## ⚙️ Operational Parameters (Features)

The system processes five core healthcare operational metrics:
* **Patients Waiting:** Current number of patients in the queue.
* **Available Doctors:** Number of active medical practitioners on duty.
* **Appointment Count:** Total scheduled appointments for the period.
* **Emergency Cases:** Number of critical/emergency cases handled.
* **Avg Consultation Time:** Average time (in minutes) spent per patient consultation.

---

## 🛠️ Software & Library Requirements

* **Programming Language:** Python 3.x
* **Libraries:** 
  * `numpy` (Numerical computations and matrix operations)
  * `pandas` (Data manipulation and structuring)

You can install the required libraries using pip:
```bash
pip install numpy pandas

🚀 Code Implementation
The repository is split into two independent modules:

Part A: Supervised Learning (Perceptron)
Utilizes a single-layer Perceptron with a step activation function.

Trains on labeled historical healthcare data over multiple epochs with automated error correction and weight updates.

Accepts custom real-time hospital inputs to predict whether the waiting condition is acceptable or excessive.

Part B: Unsupervised Learning (Competitive Learning)
Implements a competitive neural network layer with competing neurons.

Uses Euclidean distance calculations and winner-take-all weight adaptation to group similar operational states into distinct clusters.

🏃‍♂️ How to Run
1.Clone the repository or copy the code scripts.

2.Ensure you have Python and the required libraries installed.

3.Run the supervised script to train the Perceptron model and test live operational inputs:
python supervised_waiting_system.py

4.Run the unsupervised script to observe cluster formations and neuron weight adaptations:
python unsupervised_waiting_system.py


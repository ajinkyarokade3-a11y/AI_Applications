# Pharmacy Inventory and Prescription Refill Text Generation using SLM and µLM

[![Python 3.x](https://img.shields.io/badge/python-3.x-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A lightweight, educational implementation of **Small Language Models (SLMs)** and **Micro Language Models (µLMs)** tailored for pharmacy inventory management, drug dispensing instructions, and prescription refill request text generation.

---

## 🚀 Background & Motivation
Modern hospital pharmacies and outpatient dispensaries manage extensive inventories of drugs, dosage instructions, and refill requests. While Large Language Models (LLMs) can automate prescription processing, they require heavy computational infrastructure, massive cloud datasets, and introduce privacy concerns for sensitive patient pharmacological data—making them impractical for local pharmacy kiosks and low-cost inventory terminals.

To address these limitations, this project implements **µLMs** and **SLMs** built for specific pharmaceutical workflows using a compact, domain-specific text corpus.

---

## 📂 Project Structure
* **`micro_language_model.py`**: Bigram-based statistical model for single-word context prediction and text generation.
* **`small_language_model.py`**: Trigram-based statistical model for two-word context window prediction and text generation.
* **`README.md`**: Project documentation and setup guide.

---

## 🛠️ Software Requirements
* **Operating System**: Windows / Linux / macOS
* **Programming Language**: Python 3.x
* **Libraries**: `re`, `numpy`, `collections` (Standard Library)

---

## ⚙️ How It Works

### 1. Micro Language Model (µLM) – Bigram Approach
* Estimates next-word probabilities based on a single preceding pharmaceutical term using conditional probability:
  $$P(w_i\vert{}w_{i-1}) = \frac{Count(w_{i-1},w_i)} {Count(w_{i-1})}$$

* **Example Input**: `five` $\rightarrow$ **Predicted Next Word**: `hundred`

### 2. Small Language Model (SLM) – Trigram Approach
* Captures a broader context window from the previous two words to handle complex dosage structures:
  $$P(w_i\vert{}w_{i-2},w_{i-1}) = \frac{ Count(w_{i-2},w_{i-1},w_i) }{ Count(w_{i-2},w_{i-1}) }$$

* **Example Input**: `take twice` $\rightarrow$ **Predicted Next Word**: `daily`

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/pharmacy-slm-ulm.git](https://github.com/your-username/pharmacy-slm-ulm.git)
cd pharmacy-slm-ulm

#Run the Micro Language Model (µLM)
python micro_language_model.py

#Run the Small Language Model (SLM)
python small_language_model.py

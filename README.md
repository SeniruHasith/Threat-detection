# Emergency Call Stress and Incident Analysis Model

This project aims to develop a deep learning model that can analyze emergency calls by detecting stress levels and classifying incident types. The model uses audio features such as Mel-frequency cepstral coefficients (MFCCs) to predict stress and incident types based on input audio files.

---

## Table of Contents
1. [Overview](#overview)
2. [Requirements](#requirements)
3. [Setup and Installation](#setup-and-installation)
4. [Usage](#usage)
5. [Model Training](#model-training)
6. [Model Inference](#model-inference)
7. [Example](#example)
8. [License](#license)

---

## Overview

This project consists of two main components:
1. **Emergency Call Trainer**: A class responsible for training a deep learning model to predict stress levels and incident types from audio features.
2. **Emergency Call Analyzer**: A class responsible for analyzing emergency call audio and providing predictions for stress levels, incident types, and priority levels based on the trained model.

The model uses **Convolutional Neural Networks (CNNs)** and **Bidirectional Long Short-Term Memory (BiLSTM)** layers to extract audio features, followed by **Dense layers** for classification.

---

## Requirements

- Python 3.7+
- TensorFlow 2.x
- librosa
- audiomentations
- NumPy
- Scikit-learn
- Matplotlib
- scipy

You can install the necessary dependencies by running:

```bash
pip install -r requirements.txt

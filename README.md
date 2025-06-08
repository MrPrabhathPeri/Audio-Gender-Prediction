# 🎤 Audio Gender Detection

This project is a **gender classification system** that uses a neural network to predict whether a speaker is male or female based on their voice. The system uses MFCC (Mel Frequency Cepstral Coefficients) features extracted from short audio clips.

## 📁 Dataset Used

- **Free Spoken Digit Dataset (FSDD)**
- Source: [GitHub - Jakobovski/free-spoken-digit-dataset](https://github.com/Jakobovski/free-spoken-digit-dataset)

## 🚀 Features

- Audio preprocessing with silence trimming
- MFCC feature extraction
- Binary gender classification (Male / Female)
- Deep learning model trained on MFCC features
- Predictions on custom `.wav` and `.opus` audio files

## 🧠 Tech Stack

- **Python**
- **TensorFlow / Keras** – deep learning
- **Librosa** – audio processing
- **Pandas & NumPy** – data handling
- **Scikit-learn** – preprocessing and evaluation
- **Matplotlib** – plotting training history

## 📌 How It Works

1. Clone the dataset and extract `.wav` files.
2. Label each file based on the speaker name:
   - `jackson`, `nicolas` → **Male**
   - `theo` → **Female**
3. Preprocess audio:
   - Trim silence using `librosa.effects.trim()`
   - Extract MFCC features (13 coefficients)
4. Train a deep neural network on the MFCC features.
5. Save the trained model to `audio_gender_detection_model.h5`.
6. Load the model to predict gender from new audio samples.


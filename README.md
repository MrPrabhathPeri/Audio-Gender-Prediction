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

## 🛠 How to Run

1. Clone the dataset:
   ```bash
   git clone https://github.com/Jakobovski/free-spoken-digit-dataset.git
   ```

2. Install the required libraries:
   ```bash
   pip install librosa numpy tensorflow matplotlib scikit-learn
   ```

3. Open the notebook or script in your Python environment (Google Colab recommended).

4. Set the correct path to the dataset:
   ```
   /content/free-spoken-digit-dataset/recordings/
   ```

5. Run all cells to train the model and evaluate its accuracy.

6. Save the trained model:
   ```
   audio_gender_detection_model.h5
   ```

7. To make predictions:
   - Preprocess your input `.wav` or `.opus` file
   - Use the saved model to classify the gender
   - Example output:
     ```
     Prediction: male
     [[0.8945321]]
     ```

---

## 📎 To-Do / Possible Improvements

- Build a Streamlit or Flask web interface for real-time predictions
- Add support for more diverse voices and accents
- Perform hyperparameter tuning for better accuracy
- Use audio augmentation techniques for better generalization
- Try alternative features (e.g., Chroma, Spectral Contrast)
- Train on a larger, more diverse dataset beyond FSDD



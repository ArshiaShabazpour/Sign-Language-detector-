# ✋ Sign Language Gesture Recognition

This project aims to detect hand landmarks and interpret the sign language meaning of various gestures. Using OpenCV for video capture and MediaPipe for hand landmark detection, the project visualizes the detected landmarks on-screen and saves them as CSV files. A machine learning model built with Scikit-Learn, soon to be replaced by an LSTM model using TensorFlow, classifies the gestures into corresponding sign language letters or words.
## 📑 Table of Contents
- [✨ Features](#-features)
- [🚀 Setup](#-setup)
- [🔧 Usage](#-usage)
  - [🔄 Mode Selection](#mode-selection)
  - [🎥 Capture Mode](#capture-mode)
  - [🖥️ Detection Mode](#detection-mode)
- [📁 Project Structure](#-project-structure)
- [📈 Future Enhancements](#-future-enhancements)
- [🖼️ Suggested Visual Enhancements](#-suggested-visual-enhancements)
- [🙏 Acknowledgments](#-acknowledgments)

## ✨ Features
- **Mode Selection**: Choose between:
  - **Capture Mode** (press `1`): Collect training data for new signs.
  - **Detection Mode** (press `2`): Detect gestures in real-time.
  - **Current Supported Letters**: Detects **A**, **B**, and **D**; expandable with additional data.
- **Data Capture**: Capture frames and store hand landmarks as vector data in CSV format.
- **Machine Learning Model**: Scikit-Learn SVM model, with planned replacement by an LSTM model.
- **Future Update**: Incorporate frame sequences to capture gesture motion through LSTM.

## 🚀 Setup
1. **Install Dependencies**:
   ```bash
   pip install opencv-python mediapipe numpy pandas scikit-learn tensorflow
   ```
2. **Run the App**:
   ```bash
   python data_collection.py
   ```

## 🔧 Usage

### 🔄 Mode Selection
- **Press `1`**: Enter Capture Mode to collect data.
- **Press `2`**: Enter Detection Mode for real-time classification.
- **Press `Esc`**: Close the app.

### 🎥 Capture Mode
- **Select Letter**: Press the corresponding key (e.g., `A` for letter A).
- **Capture Frames**: Press `Space` to capture.
- **Delete Latest Capture**: Press `z`.
- **Save Captured Data**: Press `s` to store in `data/Landmark_data.csv`.

### 🖥️ Detection Mode
View real-time predictions of hand gestures, displayed on-screen for user feedback.

## 📁 Project Structure

- `data/Landmark_data.csv`: Stores captured training data.
- `data_collection.py`: Main script for video input, hand landmark detection, and data collection.

## 📈 Future Enhancements

- **LSTM Model Integration**: To account for motion-dependent gestures, enhancing recognition with sequential data.
- **Expanded Vocabulary**: Include words and complex gestures.

## 🖼️ Suggested Visual Enhancements

- **Screenshots**:
  - Mode Selection Screen and Detection Mode in Action.
- **GIFs**:
  - Real-time Gesture Detection.
  - Data Capture Workflow for a letter.
- **Video Tutorials**:
  - Setting Up and Running the App.
  - Real-Time Gesture Recognition Demo.

## 🙏 Acknowledgments
- **OpenCV** for video handling.
- **MediaPipe** for hand landmark detection.
- **Scikit-Learn** and **TensorFlow** for machine learning.


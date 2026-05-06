# Bridge the Gap (Sign-Connect)

A website providing features focusing on essentially bridging the gap between the deaf community and hearing individuals.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Holistic-green)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-red)

## Overview
**Bridge the Gap (Sign-Connect)** is an assistive technology project designed to perform real-time translation of Indian Sign Language (ISL) into text. By translating 70 distinct ISL gestures, this system aims to facilitate seamless communication between the deaf community and individuals who do not know sign language. The core engine is built using advanced sequence modeling in PyTorch to accurately capture the temporal dynamics of sign language.

## Key Features
* **Real-Time Translation:** Processes live webcam feeds to predict signs dynamically with visual UI feedback.
* **Grammar Structuring:** Includes a dedicated sentence buffer that converts isolated signs into grammatically correct sentences.
* **Extensive Vocabulary:** Trained to recognize 70 essential ISL gestures, including conversational staples like "Hello," "Thank you," and "How are you."
* **Robust Feature Extraction:** Utilizes MediaPipe Holistic to extract exactly 225 key body, hand, and face landmarks, stripping away background noise and focusing entirely on skeletal movement.
* **Interactive UI Controls:** Built-in controls to collect signs, undo actions, and generate complete sentences on the fly.

## Model Architecture
The gesture recognition is powered by a **Bidirectional Long Short-Term Memory (Bi-LSTM)** network built with PyTorch (`torch.nn`). 
* **Temporal Sequencing:** Unlike static image classifiers, our Bi-LSTM processes sequences of exactly **32 frames** per gesture. This captures both the forward and backward context of the movement, which is critical for accurate sign language interpretation.
* **Optimization:** Trained using the Adam Optimizer (learning rate 1e-4) to effectively manage the sparse gradients of landmark movements.

## Performance Metrics
The model was rigorously trained on a custom-compiled dataset combining baseline data with specially recorded real-world variations.

* **Training Accuracy:** ~99.8%
* **Training Loss:** ~0.03
* **Validation Accuracy:** ~91.2%
* **Validation Loss:** ~0.65

*Note: The model demonstrates highly effective real-world generalization (91%+ accuracy) while maintaining a controlled, slight overfit to the training environment to ensure maximum reliability on core conversational phrases.*

## Future Scope
* **Meeting Interface Optimization:** Advancing the integrated meeting interface to eliminate latency, ensuring lag-free, continuous translation.
* **Vocabulary Expansion:** Scaling the dataset to include a wider array of technical and everyday phrases while maintaining a simple and accessible user experience.

## Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Mahi3114/Bridge-The-Gap---Sign-Connect.git](https://github.com/Mahi3114/Bridge-The-Gap---Sign-Connect.git)
   cd Bridge-The-Gap---Sign-Connect
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   
   # On Windows use:
   venv\Scripts\activate
   
   # On Mac/Linux use:
   source venv/bin/activate
   ```

3. **Install the dependencies:**
   Ensure you have a `requirements.txt` file in your root directory containing:
   `opencv-python==4.10.0.84`, `numpy==2.0.0`, `torch==2.11.0+cpu`, and `mediapipe==0.10.9`.
   ```bash
   pip install -r requirements.txt
   
```

4. **Model Setup:**
   Ensure you have the trained model weights (`best_model.pt`) and the label map (`label_map.npy`) placed in their correct directories as referenced in the configuration section of the main script.

## Usage

Run the main live translation script:
```bash
python realtime_predict.py
```

### **Interactive Controls**
Once the webcam feed opens, use the following keyboard inputs to interact with the system:
* `SPACE` : Collect a sign (captures a 32-frame sequence)
* `G` : Convert the current buffer of collected signs into a grammatically correct sentence
* `U` : Undo the last sign added to the buffer
* `C` : Clear the entire buffer and reset the current prediction
* `Q` : Quit the application

## Author
**Mahi Ramendra Chauksey**  

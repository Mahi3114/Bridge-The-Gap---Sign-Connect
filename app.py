import threading
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import numpy as np
import cv2
import torch
import torch.nn as nn
import mediapipe as mp
import collections
import time
from isl_grammar import SignBuffer

app = Flask(__name__)
CORS(app)

# ==========================================
# ⚙️ CONFIG (100% ALIGNED WITH YOUR SCRIPT)
# ==========================================
MODEL_PATH       = r"E:\SignConnect\new_model\best_model.pt"
LABEL_PATH       = r"E:\SignConnect\new_dataset\label_map.npy"
NUM_FRAMES       = 32  # ✅ UPDATED: Back to 32 as per your script
HIDDEN_SIZE      = 256
NUM_LAYERS       = 3
DROPOUT          = 0.4
THRESHOLD        = 0.82
INFERENCE_STRIDE = 6   # Smooth inference
VOTE_WINDOW      = 5
VOTE_NEEDED      = 3
WORD_COOLDOWN    = 2.0
PAUSE_TIMEOUT    = 4.0

label_map   = np.load(LABEL_PATH, allow_pickle=True).item()
idx_to_sign = {v: k for k, v in label_map.items()}
NUM_CLASSES = len(label_map)

sliding_window = collections.deque(maxlen=NUM_FRAMES)
vote_buffer    = collections.deque(maxlen=VOTE_WINDOW)
frame_count    = 0
last_word_time = 0.0
last_hand_time = time.time()
gramified      = False
sentence       = ""
current_pred   = "Waiting for hands..."
current_conf   = "0%"
buffer         = SignBuffer(max_signs=15)

# ==========================================
# 🤖 PYTORCH MODEL
# ==========================================
class SignLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=225, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, batch_first=True, dropout=DROPOUT, bidirectional=True)
        self.bn   = nn.BatchNorm1d(HIDDEN_SIZE * 2)
        self.drop = nn.Dropout(DROPOUT)
        self.fc1  = nn.Linear(HIDDEN_SIZE * 2, 128)
        self.relu = nn.ReLU()
        self.fc2  = nn.Linear(128, NUM_CLASSES)

    def forward(self, x):
        out, _ = self.lstm(x)
        out    = self.fc2(self.relu(self.fc1(self.drop(self.bn(out[:, -1, :])))))
        return out

device = torch.device("cpu")
model  = SignLSTM().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ==========================================
# 🔑 LANDMARK EXTRACTION (THE CRITICAL FIX)
# ==========================================
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_lock = threading.Lock()

def extract_landmarks(result):
    """Matches your script order: Right Hand -> Left Hand -> Pose"""
    landmarks = []
    
    # 1. Right Hand (63 features)
    if result.right_hand_landmarks:
        for lm in result.right_hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
    else:
        landmarks.extend([0.0] * 63)

    # 2. Left Hand (63 features)
    if result.left_hand_landmarks:
        for lm in result.left_hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
    else:
        landmarks.extend([0.0] * 63)

    # 3. Pose (99 features)
    if result.pose_landmarks:
        for lm in result.pose_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
    else:
        landmarks.extend([0.0] * 99)

    return landmarks # Total 225

def hands_present(result):
    return result.right_hand_landmarks or result.left_hand_landmarks

# ==========================================
# 🚀 PREDICT ROUTE
# ==========================================
@app.route('/predict', methods=['POST'])
def predict():
    global sliding_window, vote_buffer, frame_count, last_word_time
    global last_hand_time, gramified, sentence, current_pred, current_conf, buffer

    with mp_lock:
        try:
            data = request.json
            img_bytes = base64.b64decode(data['image'].split(',')[1])
            img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            
            now = time.time()
            result = holistic.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            if hands_present(result):
                print(">>> HANDS DETECTED")
                last_hand_time = now
                gramified = False
                sliding_window.append(extract_landmarks(result))
                frame_count += 1
            else:
                print("... No hands seen")

            if len(sliding_window) == NUM_FRAMES and hands_present(result) and frame_count % INFERENCE_STRIDE == 0:
                x = torch.tensor([list(sliding_window)], dtype=torch.float32).to(device)
                with torch.no_grad():
                    conf, idx = torch.softmax(model(x), dim=1).max(1)
                
                pred = idx_to_sign.get(idx.item(), "?")
                conf_val = conf.item()

                if conf_val >= THRESHOLD:
                    vote_buffer.append(pred)
                    current_pred = pred
                    current_conf = f"{conf_val*100:.1f}%"

                    if len(vote_buffer) == VOTE_WINDOW:
                        top_sign, top_n = collections.Counter(vote_buffer).most_common(1)[0]
                        if top_n >= VOTE_NEEDED and (now - last_word_time) >= WORD_COOLDOWN:
                            buffer.add(top_sign)
                            last_word_time = now
                            sentence = ""
                        vote_buffer.clear()
                else:
                    current_pred = "Analyzing..."

            # Auto-speak Sentence on drop hands
            pause_elapsed = now - last_hand_time
            if pause_elapsed >= PAUSE_TIMEOUT and not gramified and buffer.get_raw():
                sentence = buffer.get_sentence()
                gramified = True
                buffer.clear()
                current_pred = "Waiting for hands..."

            return jsonify({
                'word': current_pred,
                'confidence': current_conf,
                'sentence': sentence,
                'status': 'success'
            })

        except Exception as e:
            print(f"CRITICAL ERROR: {e}")
            return jsonify({'error': str(e), 'status': 'failed'}), 500

if __name__ == '__main__':
    app.run(port=5000)
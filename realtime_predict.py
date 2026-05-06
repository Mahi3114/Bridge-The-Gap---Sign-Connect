# import cv2
# import numpy as np
# import torch
# import torch.nn as nn
# import mediapipe as mp
# import collections

# # ─── CONFIG ───────────────────────────────────────────────
# MODEL_PATH   = r"E:\SignConnect\new_model\best_model.pt"
# LABEL_PATH   = r"E:\SignConnect\new_dataset\label_map.npy"
# NUM_FRAMES   = 32
# HIDDEN_SIZE  = 256
# NUM_LAYERS   = 3
# DROPOUT      = 0.4
# THRESHOLD    = 0.75
# EXCLUDE      = ["morning"]   # signs to temporarily exclude
# # ──────────────────────────────────────────────────────────

# # ─── LOAD LABEL MAP ───────────────────────────────────────
# label_map   = np.load(LABEL_PATH, allow_pickle=True).item()
# idx_to_sign = {v: k for k, v in label_map.items()}
# NUM_CLASSES = len(label_map)
# print(f"Loaded {NUM_CLASSES} signs")

# # ─── MODEL ────────────────────────────────────────────────
# class SignLSTM(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout):
#         super(SignLSTM, self).__init__()
#         self.lstm = nn.LSTM(
#             input_size    = input_size,
#             hidden_size   = hidden_size,
#             num_layers    = num_layers,
#             batch_first   = True,
#             dropout       = dropout,
#             bidirectional = True
#         )
#         self.bn   = nn.BatchNorm1d(hidden_size * 2)
#         self.drop = nn.Dropout(dropout)
#         self.fc1  = nn.Linear(hidden_size * 2, 128)
#         self.relu = nn.ReLU()
#         self.fc2  = nn.Linear(128, num_classes)

#     def forward(self, x):
#         out, _ = self.lstm(x)
#         out    = out[:, -1, :]
#         out    = self.bn(out)
#         out    = self.drop(out)
#         out    = self.fc1(out)
#         out    = self.relu(out)
#         out    = self.fc2(out)
#         return out


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model  = SignLSTM(225, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES, DROPOUT).to(device)
# model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
# model.eval()
# print(f"Model loaded on {device}")

# # ─── MEDIAPIPE ────────────────────────────────────────────
# mp_holistic   = mp.solutions.holistic
# mp_drawing    = mp.solutions.drawing_utils
# mp_draw_style = mp.solutions.drawing_styles


# def extract_landmarks(result):
#     landmarks = []
#     if result.right_hand_landmarks:
#         for lm in result.right_hand_landmarks.landmark:
#             landmarks.extend([lm.x, lm.y, lm.z])
#     else:
#         landmarks.extend([0.0] * 63)
#     if result.left_hand_landmarks:
#         for lm in result.left_hand_landmarks.landmark:
#             landmarks.extend([lm.x, lm.y, lm.z])
#     else:
#         landmarks.extend([0.0] * 63)
#     if result.pose_landmarks:
#         for lm in result.pose_landmarks.landmark:
#             landmarks.extend([lm.x, lm.y, lm.z])
#     else:
#         landmarks.extend([0.0] * 99)
#     return landmarks  # 225 values


# def predict(sequence):
#     x = torch.tensor([sequence], dtype=torch.float32).to(device)
#     with torch.no_grad():
#         logits = model(x)
#         probs  = torch.softmax(logits, dim=1)
#         # Zero out excluded signs
#         for sign in EXCLUDE:
#             if sign in label_map:
#                 probs[0][label_map[sign]] = 0
#         conf, idx = probs.max(1)
#     return idx_to_sign.get(idx.item(), "?"), conf.item()


# # ─── DRAWING HELPERS ──────────────────────────────────────
# def draw_landmarks(frame, result):
#     if result.right_hand_landmarks:
#         mp_drawing.draw_landmarks(
#             frame, result.right_hand_landmarks,
#             mp_holistic.HAND_CONNECTIONS,
#             mp_draw_style.get_default_hand_landmarks_style(),
#             mp_draw_style.get_default_hand_connections_style()
#         )
#     if result.left_hand_landmarks:
#         mp_drawing.draw_landmarks(
#             frame, result.left_hand_landmarks,
#             mp_holistic.HAND_CONNECTIONS,
#             mp_draw_style.get_default_hand_landmarks_style(),
#             mp_draw_style.get_default_hand_connections_style()
#         )
#     if result.pose_landmarks:
#         mp_drawing.draw_landmarks(
#             frame, result.pose_landmarks,
#             mp_holistic.POSE_CONNECTIONS
#         )


# def draw_ui(frame, sequence, prediction, confidence, collecting):
#     h, w = frame.shape[:2]

#     # Progress bar
#     filled = int((len(sequence) / NUM_FRAMES) * w)
#     cv2.rectangle(frame, (0, h - 20), (w, h), (50, 50, 50), -1)
#     cv2.rectangle(frame, (0, h - 20), (filled, h), (0, 200, 100), -1)
#     cv2.putText(frame, f"{len(sequence)}/{NUM_FRAMES} frames",
#                 (10, h - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

#     # Status
#     status     = "● COLLECTING" if collecting else "○ READY — Press SPACE to sign"
#     status_col = (0, 200, 100) if collecting else (200, 200, 200)
#     cv2.putText(frame, status, (10, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_col, 2)

#     # Prediction
#     if prediction:
#         color = (0, 255, 0) if confidence >= THRESHOLD else (0, 165, 255)
#         cv2.putText(frame, f"{prediction.upper()}",
#                     (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.8, color, 3)
#         cv2.putText(frame, f"Confidence: {confidence*100:.1f}%",
#                     (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

#     # Controls
#     cv2.putText(frame, "SPACE=Sign  R=Reset  Q=Quit",
#                 (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

#     return frame


# # ─── MAIN LOOP ────────────────────────────────────────────
# def main():
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         print("❌ Cannot open webcam")
#         return

#     cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

#     sequence     = []
#     prediction   = ""
#     confidence   = 0.0
#     collecting   = False
#     pred_history = collections.deque(maxlen=1)

#     print("\n✅ Webcam ready!")
#     print("   SPACE = start collecting frames for a sign")
#     print("   R     = reset")
#     print("   Q     = quit\n")

#     with mp_holistic.Holistic(
#         static_image_mode        = False,
#         model_complexity         = 1,
#         min_detection_confidence = 0.5,
#         min_tracking_confidence  = 0.5
#     ) as holistic:

#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             frame  = cv2.flip(frame, 1)
#             rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             result = holistic.process(rgb)

#             draw_landmarks(frame, result)

#             if collecting:
#                 lm = extract_landmarks(result)
#                 sequence.append(lm)

#                 if len(sequence) == NUM_FRAMES:
#                     pred, conf = predict(sequence)
#                     pred_history.append(pred)
#                     prediction = collections.Counter(pred_history).most_common(1)[0][0]
#                     confidence = conf
#                     collecting = False
#                     sequence   = []

#             frame = draw_ui(frame, sequence, prediction, confidence, collecting)
#             cv2.imshow("ISL Sign Language Recognition", frame)

#             key = cv2.waitKey(1) & 0xFF
#             if key == ord('q'):
#                 break
#             elif key == ord(' '):
#                 sequence   = []
#                 collecting = True
#                 prediction = ""
#                 print("Collecting frames...")
#             elif key == ord('r'):
#                 sequence   = []
#                 collecting = False
#                 prediction = ""
#                 pred_history.clear()
#                 print("Reset.")

#     cap.release()
#     cv2.destroyAllWindows()


# if __name__ == "__main__":
#     main()

import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp
import collections
from isl_grammar import SignBuffer

# ─── CONFIG ───────────────────────────────────────────────
MODEL_PATH   = r"E:\SignConnect\new_model\best_model.pt"
LABEL_PATH   = r"E:\SignConnect\new_dataset\label_map.npy"
NUM_FRAMES   = 32
HIDDEN_SIZE  = 256
NUM_LAYERS   = 3
DROPOUT      = 0.4
THRESHOLD    = 0.75
EXCLUDE      = []    # add signs to exclude if needed e.g. ["morning"]
# ──────────────────────────────────────────────────────────

label_map   = np.load(LABEL_PATH, allow_pickle=True).item()
idx_to_sign = {v: k for k, v in label_map.items()}
NUM_CLASSES = len(label_map)
print(f"Loaded {NUM_CLASSES} signs")

class SignLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout):
        super(SignLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            dropout=dropout, bidirectional=True)
        self.bn   = nn.BatchNorm1d(hidden_size * 2)
        self.drop = nn.Dropout(dropout)
        self.fc1  = nn.Linear(hidden_size * 2, 128)
        self.relu = nn.ReLU()
        self.fc2  = nn.Linear(128, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out    = out[:, -1, :]
        out    = self.bn(out)
        out    = self.drop(out)
        out    = self.fc1(out)
        out    = self.relu(out)
        out    = self.fc2(out)
        return out


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = SignLSTM(225, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES, DROPOUT).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print(f"Model loaded on {device}")

mp_holistic   = mp.solutions.holistic
mp_drawing    = mp.solutions.drawing_utils
mp_draw_style = mp.solutions.drawing_styles


def extract_landmarks(result):
    landmarks = []
    if result.right_hand_landmarks:
        for lm in result.right_hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
    else:
        landmarks.extend([0.0] * 63)
    if result.left_hand_landmarks:
        for lm in result.left_hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
    else:
        landmarks.extend([0.0] * 63)
    if result.pose_landmarks:
        for lm in result.pose_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
    else:
        landmarks.extend([0.0] * 99)
    return landmarks


def predict(sequence):
    x = torch.tensor([sequence], dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(x)
        probs  = torch.softmax(logits, dim=1)
        for sign in EXCLUDE:
            if sign in label_map:
                probs[0][label_map[sign]] = 0
        conf, idx = probs.max(1)
    return idx_to_sign.get(idx.item(), "?"), conf.item()


def draw_landmarks(frame, result):
    if result.right_hand_landmarks:
        mp_drawing.draw_landmarks(frame, result.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_draw_style.get_default_hand_landmarks_style(),
            mp_draw_style.get_default_hand_connections_style())
    if result.left_hand_landmarks:
        mp_drawing.draw_landmarks(frame, result.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_draw_style.get_default_hand_landmarks_style(),
            mp_draw_style.get_default_hand_connections_style())
    if result.pose_landmarks:
        mp_drawing.draw_landmarks(frame, result.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS)


def draw_ui(frame, sequence, prediction, confidence, collecting, buffer, sentence):
    h, w = frame.shape[:2]

    # Progress bar
    filled = int((len(sequence) / NUM_FRAMES) * w)
    cv2.rectangle(frame, (0, h-20), (w, h), (50,50,50), -1)
    cv2.rectangle(frame, (0, h-20), (filled, h), (0,200,100), -1)
    cv2.putText(frame, f"{len(sequence)}/{NUM_FRAMES} frames",
                (10, h-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    # Status
    status     = "● COLLECTING" if collecting else "○ READY — Press SPACE to sign"
    status_col = (0,200,100) if collecting else (200,200,200)
    cv2.putText(frame, status, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_col, 2)

    # Current prediction
    if prediction:
        color = (0,255,0) if confidence >= THRESHOLD else (0,165,255)
        cv2.putText(frame, prediction.upper(),
                    (10,80), cv2.FONT_HERSHEY_SIMPLEX, 1.8, color, 3)
        cv2.putText(frame, f"Confidence: {confidence*100:.1f}%",
                    (10,115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Word buffer (signs collected so far)
    raw = buffer.get_raw()
    if raw:
        cv2.putText(frame, f"Signs: {raw}",
                    (10,150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,0), 1)

    # Grammar sentence
    if sentence:
        cv2.putText(frame, sentence,
                    (10,185), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,255), 2)

    # Controls
    cv2.putText(frame, "SPACE=Sign  G=Grammar  U=Undo  C=Clear  Q=Quit",
                (10, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180,180,180), 1)

    return frame


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    sequence     = []
    prediction   = ""
    confidence   = 0.0
    collecting   = False
    sentence     = ""
    pred_history = collections.deque(maxlen=1)
    buffer       = SignBuffer(max_signs=15)

    print("\n✅ Webcam ready!")
    print("   SPACE = collect sign")
    print("   G     = convert buffer to grammar sentence")
    print("   U     = undo last sign")
    print("   C     = clear buffer")
    print("   Q     = quit\n")

    with mp_holistic.Holistic(
        static_image_mode=False, model_complexity=1,
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as holistic:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame  = cv2.flip(frame, 1)
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = holistic.process(rgb)
            draw_landmarks(frame, result)

            if collecting:
                lm = extract_landmarks(result)
                sequence.append(lm)

                if len(sequence) == NUM_FRAMES:
                    pred, conf = predict(sequence)
                    pred_history.append(pred)
                    prediction = collections.Counter(pred_history).most_common(1)[0][0]
                    confidence = conf
                    collecting = False
                    sequence   = []
                    # Add to buffer
                    buffer.add(prediction)
                    print(f"Predicted: {prediction} ({conf*100:.1f}%)")
                    print(f"Buffer   : {buffer.get_raw()}")

            frame = draw_ui(frame, sequence, prediction, confidence,
                            collecting, buffer, sentence)
            cv2.imshow("ISL Sign Language Recognition", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                sequence   = []
                collecting = True
                prediction = ""
                print("Collecting frames...")
            elif key == ord('g'):
                sentence = buffer.get_sentence()
                print(f"Sentence : {sentence}")
            elif key == ord('u'):
                buffer.undo()
                sentence = ""
                print(f"Undo. Buffer: {buffer.get_raw()}")
            elif key == ord('c'):
                buffer.clear()
                sentence   = ""
                prediction = ""
                pred_history.clear()
                print("Cleared.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


# import cv2
# import numpy as np
# import torch
# import torch.nn as nn
# import mediapipe as mp
# import collections
# import time
# from isl_grammar import SignBuffer

# # ─── CONFIG ───────────────────────────────────────────────
# MODEL_PATH       = r"E:\SignConnect\new_model\best_model.pt"
# LABEL_PATH       = r"E:\SignConnect\new_dataset\label_map.npy"
# NUM_FRAMES       = 32
# HIDDEN_SIZE      = 256
# NUM_LAYERS       = 3
# DROPOUT          = 0.4
# THRESHOLD        = 0.82
# EXCLUDE          = []
# WORD_COOLDOWN    = 2.0
# PAUSE_TIMEOUT    = 4.0
# INFERENCE_STRIDE = 8
# VOTE_WINDOW      = 5
# VOTE_NEEDED      = 3
# # ──────────────────────────────────────────────────────────

# label_map   = np.load(LABEL_PATH, allow_pickle=True).item()
# idx_to_sign = {v: k for k, v in label_map.items()}
# NUM_CLASSES = len(label_map)
# print(f"Loaded {NUM_CLASSES} signs")


# class SignLSTM(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout):
#         super().__init__()
#         self.lstm = nn.LSTM(
#             input_size=input_size, hidden_size=hidden_size,
#             num_layers=num_layers, batch_first=True,
#             dropout=dropout, bidirectional=True)
#         self.bn   = nn.BatchNorm1d(hidden_size * 2)
#         self.drop = nn.Dropout(dropout)
#         self.fc1  = nn.Linear(hidden_size * 2, 128)
#         self.relu = nn.ReLU()
#         self.fc2  = nn.Linear(128, num_classes)

#     def forward(self, x):
#         out, _ = self.lstm(x)
#         out    = out[:, -1, :]
#         out    = self.bn(out)
#         out    = self.drop(out)
#         out    = self.fc1(out)
#         out    = self.relu(out)
#         out    = self.fc2(out)
#         return out


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model  = SignLSTM(225, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES, DROPOUT).to(device)
# model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
# model.eval()
# print(f"Model loaded on {device}")

# mp_holistic   = mp.solutions.holistic
# mp_drawing    = mp.solutions.drawing_utils
# mp_draw_style = mp.solutions.drawing_styles


# def extract_landmarks(result):
#     landmarks = []
#     for attr, size in [
#         ("right_hand_landmarks", 63),
#         ("left_hand_landmarks",  63),
#         ("pose_landmarks",       99),
#     ]:
#         lm_obj = getattr(result, attr)
#         if lm_obj:
#             for lm in lm_obj.landmark:
#                 landmarks.extend([lm.x, lm.y, lm.z])
#         else:
#             landmarks.extend([0.0] * size)
#     return landmarks  # 225 values


# def hands_present(result):
#     return (result.right_hand_landmarks is not None or
#             result.left_hand_landmarks  is not None)


# def predict(sequence):
#     x = torch.tensor([sequence], dtype=torch.float32).to(device)
#     with torch.no_grad():
#         logits = model(x)
#         probs  = torch.softmax(logits, dim=1)
#         for sign in EXCLUDE:
#             if sign in label_map:
#                 probs[0][label_map[sign]] = 0
#         conf, idx = probs.max(1)
#     return idx_to_sign.get(idx.item(), "?"), conf.item()


# def draw_landmarks(frame, result):
#     for attr in ["right_hand_landmarks", "left_hand_landmarks"]:
#         lm_obj = getattr(result, attr)
#         if lm_obj:
#             mp_drawing.draw_landmarks(
#                 frame, lm_obj,
#                 mp_holistic.HAND_CONNECTIONS,
#                 mp_draw_style.get_default_hand_landmarks_style(),
#                 mp_draw_style.get_default_hand_connections_style())
#     if result.pose_landmarks:
#         mp_drawing.draw_landmarks(
#             frame, result.pose_landmarks,
#             mp_holistic.POSE_CONNECTIONS)


# def draw_ui(frame, sliding_window, prediction, confidence,
#             buffer, sentence, pause_elapsed, gramified,
#             vote_buffer):
#     h, w = frame.shape[:2]

#     # ── Sliding window fill bar (top strip) ──
#     filled = int((len(sliding_window) / NUM_FRAMES) * w)
#     cv2.rectangle(frame, (0, 0), (w, 6), (50, 50, 50), -1)
#     cv2.rectangle(frame, (0, 0), (filled, 6), (0, 200, 100), -1)

#     # ── Vote progress dots ──
#     dot_x = 10
#     for i in range(VOTE_WINDOW):
#         filled_dot = i < len(vote_buffer)
#         color = (0, 220, 120) if filled_dot else (80, 80, 80)
#         cv2.circle(frame, (dot_x + i * 18, 20), 6, color, -1)
#     cv2.putText(frame, "vote",
#                 (dot_x + VOTE_WINDOW * 18 + 6, 24),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 140), 1)

#     # ── Current prediction ──
#     if prediction:
#         color = (0, 255, 0) if confidence >= THRESHOLD else (0, 165, 255)
#         cv2.putText(frame, prediction.upper(),
#                     (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.8, color, 3)
#         cv2.putText(frame, f"{confidence*100:.1f}%",
#                     (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

#     # ── Word buffer ──
#     raw = buffer.get_raw()
#     if raw:
#         cv2.putText(frame, f"Signs: {raw}",
#                     (10, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 0), 1)

#     # ── Grammar sentence ──
#     if sentence:
#         cv2.putText(frame, sentence,
#                     (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

#     # ── Pause countdown bar ──
#     if pause_elapsed > 0.5 and not gramified and raw:
#         ratio   = min(pause_elapsed / PAUSE_TIMEOUT, 1.0)
#         bar_w   = int(ratio * (w - 20))
#         r       = int(255 * ratio)
#         g       = int(200 * (1 - ratio))
#         cv2.rectangle(frame, (10, h - 50), (w - 10, h - 38), (50, 50, 50), -1)
#         cv2.rectangle(frame, (10, h - 50), (10 + bar_w, h - 38), (r, g, 0), -1)
#         secs_left = max(0.0, PAUSE_TIMEOUT - pause_elapsed)
#         cv2.putText(frame, f"Auto-gramify in {secs_left:.1f}s",
#                     (10, h - 53), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)

#     # ── Controls ──
#     cv2.putText(frame, "G=Grammar  U=Undo  C=Clear  Q=Quit",
#                 (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1)

#     return frame


# def main():
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         print("❌ Cannot open webcam")
#         return
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

#     sliding_window = collections.deque(maxlen=NUM_FRAMES)
#     vote_buffer    = collections.deque(maxlen=VOTE_WINDOW)

#     prediction     = ""
#     confidence     = 0.0
#     sentence       = ""
#     gramified      = False

#     frame_count    = 0
#     last_word_time = 0.0
#     last_hand_time = time.time()
#     pause_elapsed  = 0.0

#     buffer = SignBuffer(max_signs=15)

#     print("\n✅ Running — just start signing!")
#     print("   G = manual gramify  |  U = undo  |  C = clear  |  Q = quit\n")

#     with mp_holistic.Holistic(
#         static_image_mode=False,
#         model_complexity=1,
#         min_detection_confidence=0.5,
#         min_tracking_confidence=0.5
#     ) as holistic:

#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             now    = time.time()
#             frame  = cv2.flip(frame, 1)
#             rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             result = holistic.process(rgb)
#             draw_landmarks(frame, result)

#             # ── Hand presence & pause tracking ──
#             if hands_present(result):
#                 last_hand_time = now
#                 pause_elapsed  = 0.0
#                 gramified      = False
#             else:
#                 pause_elapsed = now - last_hand_time

#             # ── Feed sliding window every frame ──
#             lm = extract_landmarks(result)
#             sliding_window.append(lm)
#             frame_count += 1

#             # ── Inference on stride, only when hands present & window full ──
#             if (len(sliding_window) == NUM_FRAMES
#                     and hands_present(result)
#                     and frame_count % INFERENCE_STRIDE == 0):

#                 pred, conf = predict(list(sliding_window))

#                 if conf >= THRESHOLD:
#                     vote_buffer.append(pred)

#                     # Show live candidate even before commit
#                     prediction = pred
#                     confidence = conf

#                     # Majority vote reached → commit word
#                     if len(vote_buffer) == VOTE_WINDOW:
#                         counter          = collections.Counter(vote_buffer)
#                         top_sign, top_n  = counter.most_common(1)[0]

#                         if top_n >= VOTE_NEEDED:
#                             time_since_last = now - last_word_time

#                             if time_since_last >= WORD_COOLDOWN:
#                                 buffer.add(top_sign)
#                                 last_word_time = now
#                                 sentence       = ""
#                                 print(f"+ {top_sign} ({conf*100:.1f}%)  →  {buffer.get_raw()}")

#                             # Always clear after vote resolves
#                             vote_buffer.clear()
#                             sliding_window.clear()
#                             frame_count = 0

#                 else:
#                     # Low confidence — break the voting streak
#                     vote_buffer.clear()

#             # ── Auto-gramify after pause ──
#             if (pause_elapsed >= PAUSE_TIMEOUT
#                     and not gramified
#                     and buffer.get_raw()):
#                 sentence  = buffer.get_sentence()
#                 gramified = True
#                 print(f"[Auto] Sentence: {sentence}")

#             frame = draw_ui(frame, sliding_window, prediction, confidence,
#                             buffer, sentence, pause_elapsed, gramified,
#                             vote_buffer)
#             cv2.imshow("ISL Sign Language Recognition", frame)

#             key = cv2.waitKey(1) & 0xFF
#             if key == ord('q'):
#                 break
#             elif key == ord('g'):
#                 sentence  = buffer.get_sentence()
#                 gramified = True
#                 print(f"[Manual] Sentence: {sentence}")
#             elif key == ord('u'):
#                 buffer.undo()
#                 sentence  = ""
#                 gramified = False
#                 print(f"Undo. Buffer: {buffer.get_raw()}")
#             elif key == ord('c'):
#                 buffer.clear()
#                 sentence   = ""
#                 prediction = ""
#                 gramified  = False
#                 vote_buffer.clear()
#                 sliding_window.clear()
#                 frame_count = 0
#                 print("Cleared.")

#     cap.release()
#     cv2.destroyAllWindows()


# if __name__ == "__main__":
#     main()
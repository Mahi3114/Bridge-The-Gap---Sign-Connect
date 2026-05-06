import os
import cv2
import numpy as np
import mediapipe as mp

# =========================
# ⚙️ CONFIG
# =========================
DATA_PATH = r"E:\SignConnect\wlasl_videos"
SAVE_PATH = "processed_data"
SEQUENCE_LENGTH = 30

# =========================
# 🤖 MEDIAPIPE
# =========================
mp_holistic = mp.solutions.holistic

# =========================
# 🔑 EXTRACT KEYPOINTS (225)
# =========================
def extract_keypoints(results):

    # Pose (33 × 3 = 99)
    pose = np.array(
        [[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]
    ).flatten() if results.pose_landmarks else np.zeros(33 * 3)

    # Left hand (21 × 3 = 63)
    lh = np.array(
        [[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]
    ).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)

    # Right hand (21 × 3 = 63)
    rh = np.array(
        [[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]
    ).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)

    # FINAL = 225 features
    return np.concatenate([pose, lh, rh])


# =========================
# 🔧 FIX SEQUENCE LENGTH
# =========================
def fix_sequence_length(sequence, target_len=30):

    if len(sequence) == 0:
        return None

    # Too long → sample evenly
    if len(sequence) > target_len:
        indices = np.linspace(0, len(sequence) - 1, target_len).astype(int)
        sequence = sequence[indices]

    # Too short → pad
    elif len(sequence) < target_len:
        pad = np.zeros((target_len - len(sequence), sequence.shape[1]))
        sequence = np.vstack([sequence, pad])

    return sequence


# =========================
# 🎥 PROCESS VIDEO
# =========================
def process_video(video_path):

    cap = cv2.VideoCapture(video_path)
    frames = []

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame is None:
                continue

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)

            keypoints = extract_keypoints(results)
            frames.append(keypoints)

    cap.release()

    if len(frames) == 0:
        return None

    frames = np.array(frames, dtype=np.float32)

    # 🔥 FIX LENGTH HERE
    frames = fix_sequence_length(frames, SEQUENCE_LENGTH)

    return frames


# =========================
# 🚀 MAIN
# =========================
def main():

    # Safety check
    if not os.path.exists(DATA_PATH):
        print("❌ Dataset path not found")
        return

    os.makedirs(SAVE_PATH, exist_ok=True)

    actions = [
        a for a in os.listdir(DATA_PATH)
        if os.path.isdir(os.path.join(DATA_PATH, a))
    ]

    print(f"\n🧾 Total classes found: {len(actions)}")

    for action in actions:

        print(f"\n📂 Processing: {action}")

        action_path = os.path.join(DATA_PATH, action)
        sequences = []

        videos = [v for v in os.listdir(action_path) if v.endswith(".mp4")]

        for video in videos:

            video_path = os.path.join(action_path, video)

            try:
                seq = process_video(video_path)

                if seq is not None:
                    sequences.append(seq)
                else:
                    print(f"⚠️ Skipped (empty): {video}")

            except Exception as e:
                print(f"❌ Error in {video}: {e}")

        # =========================
        # 💾 SAVE CLEAN DATA
        # =========================
        if len(sequences) > 0:

            sequences = np.array(sequences, dtype=np.float32)

            save_file = os.path.join(SAVE_PATH, f"{action}.npy")
            np.save(save_file, sequences)

            print(f"✅ Saved: {action} | shape: {sequences.shape}")

        else:
            print(f"❌ No valid sequences for {action}")


# =========================
# ▶️ RUN
# =========================
if __name__ == "__main__":
    main()
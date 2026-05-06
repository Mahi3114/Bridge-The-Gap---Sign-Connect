import os
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm

# ─── CONFIG ───────────────────────────────────────────────
ORIGINAL_DATASET = r"E:\SignConnect\ISL_DATASET"   # original .MOV videos
MY_DATASET       = r"E:\SignConnect\my_DATASET"    # your .npy recordings
OUTPUT_PATH      = r"E:\SignConnect\new_dataset"   # combined output
NUM_FRAMES       = 32
# ──────────────────────────────────────────────────────────

os.makedirs(OUTPUT_PATH, exist_ok=True)
mp_holistic = mp.solutions.holistic


def extract_landmarks_from_video(video_path):
    cap    = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if len(frames) == 0:
        return None

    indices = np.linspace(0, len(frames) - 1, NUM_FRAMES, dtype=int)
    frames  = [frames[i] for i in indices]

    sequence = []
    with mp_holistic.Holistic(
        static_image_mode=False, model_complexity=1,
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as holistic:
        for frame in frames:
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = holistic.process(rgb)
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

            sequence.append(landmarks)

    arr = np.array(sequence, dtype=np.float32)
    return arr if arr.shape == (NUM_FRAMES, 225) else None


def main():
    X, y   = [], []
    failed = []

    # Combine signs from both datasets
    isl_signs = sorted(os.listdir(ORIGINAL_DATASET))
    my_signs  = sorted(os.listdir(MY_DATASET)) if os.path.exists(MY_DATASET) else []
    all_signs = sorted(set(isl_signs) | set(my_signs))
    label_map = {sign: idx for idx, sign in enumerate(all_signs)}
    signs     = isl_signs  # still only process MOV from ISL
    print(f"Found {len(all_signs)} total sign classes ({len(signs)} with original videos)\n")

    # ── PART 1: Original .MOV videos ──────────────────────
    print("=" * 50)
    print("PART 1: Processing original .MOV videos")
    print("=" * 50)

    for sign in signs:
        sign_path = os.path.join(ORIGINAL_DATASET, sign)
        if not os.path.isdir(sign_path):
            continue
        videos = [f for f in os.listdir(sign_path)
                  if f.lower().endswith(('.mov', '.mp4', '.avi'))]
        if not videos:
            continue

        print(f"Processing '{sign}' ({len(videos)} videos)...")
        for video_file in tqdm(videos, desc=sign):
            landmarks = extract_landmarks_from_video(
                os.path.join(sign_path, video_file))
            if landmarks is None:
                failed.append(video_file)
                continue
            X.append(landmarks)
            y.append(label_map[sign])

    part1_count = len(X)
    print(f"\n✅ Part 1 done — {part1_count} samples from original dataset")

    # ── PART 2: Your .npy recordings ──────────────────────
    print("\n" + "=" * 50)
    print("PART 2: Loading your personal .npy recordings")
    print("=" * 50)

    my_count = 0
    if os.path.exists(MY_DATASET):
        for sign in sorted(os.listdir(MY_DATASET)):
            sign_path = os.path.join(MY_DATASET, sign)
            if not os.path.isdir(sign_path):
                continue
            if sign not in label_map:
                print(f"  ⚠️  '{sign}' not in label map — skipping")
                continue
            npy_files = [f for f in os.listdir(sign_path) if f.endswith('.npy')]
            for npy_file in npy_files:
                landmarks = np.load(os.path.join(sign_path, npy_file))
                if landmarks.shape != (NUM_FRAMES, 225):
                    continue
                X.append(landmarks)
                y.append(label_map[sign])
                my_count += 1
        print(f"✅ Part 2 done — {my_count} samples from your recordings")
    else:
        print(f"⚠️  {MY_DATASET} not found")

    # ── SAVE ──────────────────────────────────────────────
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    np.save(os.path.join(OUTPUT_PATH, "X_landmarks.npy"), X)
    np.save(os.path.join(OUTPUT_PATH, "y_labels.npy"),    y)
    np.save(os.path.join(OUTPUT_PATH, "label_map.npy"),   label_map)

    print(f"\n{'='*50}")
    print(f"✅ DONE!")
    print(f"   Original samples : {part1_count}")
    print(f"   Your samples     : {my_count}")
    print(f"   Total samples    : {len(X)}")
    print(f"   X shape          : {X.shape}")
    print(f"   Saved to         : {OUTPUT_PATH}")

    if failed:
        print(f"\n⚠️  {len(failed)} videos failed:")
        for f in failed:
            print(f"   {f}")


if __name__ == "__main__":
    main()
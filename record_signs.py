import cv2
import numpy as np
import mediapipe as mp
import os

# ─── CONFIG ───────────────────────────────────────────────
DATASET_PATH       = r"E:\SignConnect\ISL_DATASET"
MY_DATA_PATH       = r"E:\SignConnect\my_DATASET"
NUM_FRAMES         = 32
VIDEOS_PER_SIGN    = 25    # 10 already done + 15 more = 25 total  
# NEW_SIGNS          = ["my", "name_is", "hello", "thank_you", "sorry", "how_are_you", "help"]
NEW_SIGNS          = ["Namaste","I_am_fine", "Bye_Bye", "You"]  # add your new signs here
NEW_SIGN_COUNT     = 45    # 45 videos for brand new signs
# ──────────────────────────────────────────────────────────

os.makedirs(MY_DATA_PATH, exist_ok=True)
    
# Build sign list — existing + new  
existing_signs = sorted(os.listdir(DATASET_PATH))
all_signs      = existing_signs.copy()
for s in NEW_SIGNS:
    if s not in all_signs:
        all_signs.append(s)
        os.makedirs(os.path.join(MY_DATA_PATH, s), exist_ok=True)

def target_count(sign):
    return NEW_SIGN_COUNT if sign in NEW_SIGNS else VIDEOS_PER_SIGN

print(f"Total signs: {len(all_signs)}")
print(f"Existing signs: {VIDEOS_PER_SIGN} videos each (10 done + 15 more)")
print(f"New signs {NEW_SIGNS}: {NEW_SIGN_COUNT} videos each\n")

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


def put_center(frame, text, y, scale=1.0, color=(255,255,255), thickness=2):
    h, w = frame.shape[:2]
    size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0]
    x    = (w - size[0]) // 2
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Resume from where we left off
    sign_idx = 0
    for i, sign in enumerate(all_signs):
        sign_folder = os.path.join(MY_DATA_PATH, sign)
        if os.path.exists(sign_folder):
            existing = len([f for f in os.listdir(sign_folder) if f.endswith('.npy')])
            if existing >= target_count(sign):
                sign_idx = i + 1
            else:
                break

    if sign_idx > 0 and sign_idx < len(all_signs):
        print(f"Resuming from: {all_signs[sign_idx]}")
    elif sign_idx >= len(all_signs):
        print("All signs already recorded!")
        return

    with mp_holistic.Holistic(
        static_image_mode=False, model_complexity=1,
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as holistic:

        while sign_idx < len(all_signs):
            sign        = all_signs[sign_idx]
            sign_folder = os.path.join(MY_DATA_PATH, sign)
            os.makedirs(sign_folder, exist_ok=True)
            needed      = target_count(sign)

            existing  = [f for f in os.listdir(sign_folder) if f.endswith('.npy')]
            video_idx = len(existing)

            if video_idx >= needed:
                sign_idx += 1
                continue

            while video_idx < needed:
                state    = "waiting"
                sequence = []

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame  = cv2.flip(frame, 1)
                    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result = holistic.process(rgb)
                    draw_landmarks(frame, result)

                    h, w = frame.shape[:2]

                    if state == "waiting":
                        color = (255, 100, 0) if sign in NEW_SIGNS else (0, 255, 255)
                        put_center(frame, f"{sign.upper()}", 70,
                                   scale=2.0, color=color, thickness=3)
                        if sign in NEW_SIGNS:
                            put_center(frame, "★ NEW SIGN", 115,
                                       scale=0.8, color=(255, 100, 0))
                        put_center(frame, f"Video {video_idx+1} of {needed}", 150,
                                   scale=0.8, color=(200, 200, 200))
                        put_center(frame, "Press SPACE when ready to sign", 210,
                                   scale=0.8, color=(255, 255, 255))

                        total_done = sum(
                            min(len([f for f in os.listdir(os.path.join(MY_DATA_PATH, s))
                                     if f.endswith('.npy')]), target_count(s))
                            if os.path.exists(os.path.join(MY_DATA_PATH, s)) else 0
                            for s in all_signs)
                        total_all = sum(target_count(s) for s in all_signs)
                        pct = int((total_done / total_all) * w)
                        cv2.rectangle(frame, (0, h-20), (w, h), (50,50,50), -1)
                        cv2.rectangle(frame, (0, h-20), (pct, h), (0, 180, 255), -1)
                        cv2.putText(frame, f"{total_done}/{total_all} total",
                                    (10, h-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                        cv2.putText(frame, "SPACE=Record  S=Skip  Q=Quit",
                                    (10, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1)

                    elif state == "recording":
                        lm = extract_landmarks(result)
                        sequence.append(lm)

                        filled = int((len(sequence) / NUM_FRAMES) * w)
                        cv2.rectangle(frame, (0, h-20), (w, h), (50,50,50), -1)
                        cv2.rectangle(frame, (0, h-20), (filled, h), (0, 255, 100), -1)
                        cv2.putText(frame, f"Recording... {len(sequence)}/{NUM_FRAMES}",
                                    (10, h-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                        put_center(frame, f"{sign.upper()}", 70,
                                   scale=2.0, color=(0, 255, 100), thickness=3)
                        put_center(frame, "● RECORDING", 130,
                                   scale=1.0, color=(0, 255, 100), thickness=2)

                        if len(sequence) == NUM_FRAMES:
                            save_path = os.path.join(sign_folder, f"{video_idx:03d}.npy")
                            np.save(save_path, np.array(sequence, dtype=np.float32))
                            print(f"  ✅ Saved: {sign} video {video_idx+1}/{needed}")
                            video_idx += 1
                            break

                    cv2.imshow("ISL Recorder", frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\nQuitting — progress saved!")
                        cap.release()
                        cv2.destroyAllWindows()
                        return
                    elif key == ord(' ') and state == "waiting":
                        state    = "recording"
                        sequence = []
                    elif key == ord('s'):
                        print(f"  ⏭ Skipping: {sign}")
                        video_idx = needed
                        break

            sign_idx += 1

        print("\n🎉 All signs recorded!")
        print(f"Data saved to: {MY_DATA_PATH}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
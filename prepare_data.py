import os
import numpy as np
from collections import Counter
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

# ==============================
# Config
# ==============================
DATASET_PATH = "dataset"
SEQUENCE_LENGTH = 30
FEATURE_LENGTH = 126
MIN_SAMPLES = 20   # remove weak classes

# ==============================
# Load dataset
# ==============================
X = []
y = []

labels = os.listdir(DATASET_PATH)

print("📂 Loading dataset...")

for label in labels:
    class_path = os.path.join(DATASET_PATH, label)

    if not os.path.isdir(class_path):
        continue

    for file in tqdm(os.listdir(class_path), desc=f"Processing {label}"):

        file_path = os.path.join(class_path, file)

        try:
            sequence = np.load(file_path, allow_pickle=True)

            # ✅ Ensure correct shape
            if sequence.shape != (SEQUENCE_LENGTH, FEATURE_LENGTH):
                print(f"Skipping invalid file: {file_path}")
                continue

            X.append(sequence)
            y.append(label)

        except Exception as e:
            print(f"Skipping corrupted file: {file_path}")
            continue

# Convert to numpy
X = np.array(X)
y = np.array(y)

print("\n📊 Initial Data Shape:", X.shape)

# ==============================
# Remove weak classes
# ==============================
print("\n⚖️ Filtering weak classes...")

counter = Counter(y)

valid_classes = [cls for cls, count in counter.items() if count >= MIN_SAMPLES]

print("Classes kept:", len(valid_classes))

indices = [i for i in range(len(y)) if y[i] in valid_classes]

X = X[indices]
y = y[indices]

print("After filtering:", X.shape)

# ==============================
# Encode labels
# ==============================
print("\n🔠 Encoding labels...")

le = LabelEncoder()
y_encoded = le.fit_transform(y)

# ==============================
# Save everything
# ==============================
os.makedirs("model", exist_ok=True)

np.save("model/X_clean.npy", X)
np.save("model/y_encoded.npy", y_encoded)
np.save("model/classes.npy", le.classes_)

print("\n📊 Final Class Distribution:")
print(Counter(y))

print("\n💾 Files saved:")
print("✔ model/X_clean.npy")
print("✔ model/y_encoded.npy")
print("✔ model/classes.npy")

print("\n✅ Data preparation COMPLETE!")
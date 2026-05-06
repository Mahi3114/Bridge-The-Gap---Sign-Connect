import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from collections import Counter
import matplotlib.pyplot as plt
import os

# ─── CONFIG ───────────────────────────────────────────────
DATA_PATH    = r"E:\SignConnect\new_dataset"
OUTPUT_PATH  = r"E:\SignConnect\new_model"
BATCH_SIZE   = 32
EPOCHS       = 120
LR           = 0.001
HIDDEN_SIZE  = 256
NUM_LAYERS   = 3
DROPOUT      = 0.4
EARLY_STOP   = 15     # stop if no improvement for 15 epochs
# ──────────────────────────────────────────────────────────

os.makedirs(OUTPUT_PATH, exist_ok=True)

# ─── LOAD DATA ────────────────────────────────────────────
print("Loading data...")
X         = np.load(os.path.join(DATA_PATH, "X_landmarks.npy"))
y         = np.load(os.path.join(DATA_PATH, "y_labels.npy"))
label_map = np.load(os.path.join(DATA_PATH, "label_map.npy"), allow_pickle=True).item()

NUM_CLASSES = len(label_map)
print(f"  X shape      : {X.shape}")
print(f"  y shape      : {y.shape}")
print(f"  Num classes  : {NUM_CLASSES}")

# ─── DATASET WITH AUGMENTATION ────────────────────────────
class LandmarkDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X       = torch.tensor(X, dtype=torch.float32)
        self.y       = torch.tensor(y, dtype=torch.long)
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].clone()
        if self.augment:
            # Add small random noise to landmarks
            x = x + torch.randn_like(x) * 0.01
            # Random temporal shift
            shift = np.random.randint(-2, 3)
            if shift > 0:
                x = torch.cat([x[shift:], x[-shift:]], dim=0)
            elif shift < 0:
                x = torch.cat([x[:shift], x[shift:]], dim=0)
        return x, self.y[idx]


# Train / val / test split: 70 / 15 / 15
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42
)

print(f"\n  Train samples: {len(X_train)}")
print(f"  Val   samples: {len(X_val)}")
print(f"  Test  samples: {len(X_test)}\n")

# augment=True only for training
train_loader = DataLoader(LandmarkDataset(X_train, y_train, augment=True), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(LandmarkDataset(X_val,   y_val,   augment=False), batch_size=BATCH_SIZE)
test_loader  = DataLoader(LandmarkDataset(X_test,  y_test,  augment=False), batch_size=BATCH_SIZE)

# ─── MODEL ────────────────────────────────────────────────
class SignLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout):
        super(SignLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size    = input_size,
            hidden_size   = hidden_size,
            num_layers    = num_layers,
            batch_first   = True,
            dropout       = dropout,
            bidirectional = True
        )

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
print(f"Using device: {device}\n")

model = SignLSTM(
    input_size  = 225,
    hidden_size = HIDDEN_SIZE,
    num_layers  = NUM_LAYERS,
    num_classes = NUM_CLASSES,
    dropout     = DROPOUT
).to(device)

print(model)
print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}\n")

# ─── TRAINING ─────────────────────────────────────────────
# Class weights to handle imbalance
counts  = Counter(y_train.tolist())
total   = len(y_train)
weights = torch.zeros(NUM_CLASSES)
for idx, count in counts.items():
    weights[idx] = total / (NUM_CLASSES * count)
weights   = weights.to(device)
criterion = nn.CrossEntropyLoss(weight=weights)

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

train_losses, val_losses = [], []
train_accs,   val_accs   = [], []
best_val_acc     = 0.0
patience_counter = 0


def run_epoch(loader, train=True):
    if train:
        model.train()
    else:
        model.eval()

    total_loss, correct, total = 0, 0, 0

    with torch.set_grad_enabled(train):
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            logits = model(X_batch)
            loss   = criterion(logits, y_batch)

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * len(y_batch)
            correct    += (logits.argmax(1) == y_batch).sum().item()
            total      += len(y_batch)

    return total_loss / total, correct / total


print("=" * 55)
print(f"{'Epoch':>6} {'Train Loss':>11} {'Train Acc':>10} {'Val Loss':>9} {'Val Acc':>9}")
print("=" * 55)

for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_acc = run_epoch(train_loader, train=True)
    vl_loss, vl_acc = run_epoch(val_loader,   train=False)

    scheduler.step(vl_loss)

    train_losses.append(tr_loss)
    val_losses.append(vl_loss)
    train_accs.append(tr_acc)
    val_accs.append(vl_acc)

    # Save best model + early stopping
    if vl_acc > best_val_acc:
        best_val_acc     = vl_acc
        patience_counter = 0
        torch.save(model.state_dict(), os.path.join(OUTPUT_PATH, "best_model.pt"))
        flag = " ← best"
    else:
        patience_counter += 1
        flag = ""

    if epoch % 5 == 0 or epoch == 1:
        print(f"{epoch:>6} {tr_loss:>11.4f} {tr_acc*100:>9.2f}% {vl_loss:>9.4f} {vl_acc*100:>8.2f}%{flag}")

    if patience_counter >= EARLY_STOP:
        print(f"\nEarly stopping at epoch {epoch} — no improvement for {EARLY_STOP} epochs")
        break

print("=" * 55)
print(f"\n✅ Best val accuracy: {best_val_acc*100:.2f}%")

# ─── EVALUATE ON TEST SET ─────────────────────────────────
print("\nEvaluating on test set...")
model.load_state_dict(torch.load(os.path.join(OUTPUT_PATH, "best_model.pt")))

all_preds, all_labels = [], []
model.eval()
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        preds   = model(X_batch).argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y_batch.numpy())

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, zero_division=0))

# ─── PLOT ─────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(train_losses, label="Train Loss")
ax1.plot(val_losses,   label="Val Loss")
ax1.set_title("Loss")
ax1.set_xlabel("Epoch")
ax1.legend()

ax2.plot([a*100 for a in train_accs], label="Train Acc")
ax2.plot([a*100 for a in val_accs],   label="Val Acc")
ax2.set_title("Accuracy (%)")
ax2.set_xlabel("Epoch")
ax2.legend()

plt.tight_layout()
plot_path = os.path.join(OUTPUT_PATH, "training_curves.png")
plt.savefig(plot_path)
plt.show()
print(f"\nPlot saved to {plot_path}")
print(f"Model saved to {os.path.join(OUTPUT_PATH, 'best_model.pt')}")
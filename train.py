from dataset import dataset
from model import model
from sklearn.model_selection import StratifiedKFold
import numpy as np
import glob
import os


MODELS_DIR = "models"


def next_model_number():
    os.makedirs(MODELS_DIR, exist_ok=True)
    existing = glob.glob(os.path.join(MODELS_DIR, "model*.json"))
    nums = []
    for f in existing:
        num_part = os.path.splitext(os.path.basename(f))[0][5:]
        if num_part.isdigit():
            nums.append(int(num_part))
    return max(nums, default=0) + 1

print("Loading dataset...")
data = dataset()

X = np.array(data.x_all)
y = np.array(data.y_all)

N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

print(f"\nRunning {N_SPLITS}-fold cross-validation...\n")
print(f"{'Fold':<6} {'Train Acc':>10} {'Val Acc':>10}")
print("-" * 28)

fold_train_accs = []
fold_val_accs = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    x_train, x_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    m = model()
    m.train(x_train, y_train, x_val, y_val)

    train_acc = m.score(x_train, y_train)
    val_acc = m.score(x_val, y_val)
    fold_train_accs.append(train_acc)
    fold_val_accs.append(val_acc)

    print(f"{fold:<6} {train_acc:>10.4f} {val_acc:>10.4f}")

print("-" * 28)
print(f"{'Mean':<6} {np.mean(fold_train_accs):>10.4f} {np.mean(fold_val_accs):>10.4f}")
print(f"{'Std':<6} {np.std(fold_train_accs):>10.4f} {np.std(fold_val_accs):>10.4f}")

answer = input("\nTrain final model on all data and save? (y/n): ").strip().lower()
if answer == "y":
    n = next_model_number()
    filename = os.path.join(MODELS_DIR, f"model{n:02d}.json")
    print(f"\nTraining final model on full dataset...")
    final_model = model()
    # Hold out last 20% for early stopping — not used for scoring, just stopping
    split = int(len(X) * 0.8)
    final_model.train(X[:split], y[:split], X[split:], y[split:])
    final_model.save(filename)
    print(f"Saved to {filename}")
else:
    print("Model not saved.")

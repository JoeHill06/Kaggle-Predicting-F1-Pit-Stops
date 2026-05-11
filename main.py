from dataset import dataset
from model import model
import numpy as np
import glob
import os


MODELS_DIR = "models"


def list_models():
    files = sorted(glob.glob(os.path.join(MODELS_DIR, "model*.json")))
    return [f for f in files if os.path.splitext(os.path.basename(f))[0][5:].isdigit()]


def pick_model(models):
    if not models:
        print("No saved models found. Run train.py first.")
        return None
    print("\nAvailable models:")
    for i, f in enumerate(models, 1):
        print(f"  {i}. {f}")
    choice = input(f"\nSelect model [1-{len(models)}] (Enter for latest): ").strip()
    if choice == "":
        return models[-1]
    if choice.isdigit() and 1 <= int(choice) <= len(models):
        return models[int(choice) - 1]
    print("Invalid choice.")
    return None


models = list_models()
model_file = pick_model(models)
if model_file is None:
    exit(1)

# Derive matching submission number from the model filename (e.g. model03.json -> submission03.csv)
num_part = os.path.splitext(os.path.basename(model_file))[0][5:]
submission_file = f"submission{num_part}.csv"

print(f"\nLoading dataset...")
data = dataset()

print(f"Loading {model_file}...")
loaded = model()
loaded.load(model_file)

print("Generating predictions...")
predictions = loaded.predict(np.array(data.x_test))

with open(submission_file, "w") as f:
    f.write("id,PitNextLap\n")
    for row_id, pred in zip(data.test_ids, predictions):
        f.write(f"{row_id},{int(pred)}\n")

print(f"Saved {len(predictions)} predictions to {submission_file}")

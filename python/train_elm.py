import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay


# ============================================================
# LOAD PREPARED DATASET
# ============================================================
DATA_DIR = Path("DATASET_TRANSFO_TEST")
NPZ_PATH = DATA_DIR / "elm_dataset_ready.npz"

data = np.load(NPZ_PATH, allow_pickle=True)

X_train = data["X_train"].astype(np.float32)
y_train = data["y_train"].astype(np.int64)

X_val = data["X_val"].astype(np.float32)
y_val = data["y_val"].astype(np.int64)

X_test = data["X_test"].astype(np.float32)
y_test = data["y_test"].astype(np.int64)

feature_names = data["feature_names"]


# ============================================================
# RECOVER CLASS LABELS
# Compatible with old and new NPZ versions
# ============================================================
if "labels_original" in data.files:
    original_labels = data["labels_original"].astype(np.int64)
elif "labels" in data.files:
    original_labels = data["labels"].astype(np.int64)
else:
    original_labels = np.sort(np.unique(np.concatenate([y_train, y_val, y_test]))).astype(np.int64)

print("Shapes:")
print("X_train :", X_train.shape)
print("X_val   :", X_val.shape)
print("X_test  :", X_test.shape)
print("Detected labels :", original_labels)
print("Number of features :", len(feature_names))


# ============================================================
# ROBUST REMAPPING OF LABELS TO 0..C-1 IF NEEDED
# ============================================================
all_y = np.concatenate([y_train, y_val, y_test])
unique_y = np.sort(np.unique(all_y))

if np.array_equal(unique_y, np.arange(len(unique_y))):
    y_train_enc = y_train.copy()
    y_val_enc = y_val.copy()
    y_test_enc = y_test.copy()

    if len(original_labels) == len(unique_y):
        class_names = [str(int(x)) for x in original_labels]
    else:
        class_names = [str(int(x)) for x in unique_y]

else:
    label_to_index = {lab: i for i, lab in enumerate(unique_y)}

    y_train_enc = np.array([label_to_index[v] for v in y_train], dtype=np.int64)
    y_val_enc   = np.array([label_to_index[v] for v in y_val], dtype=np.int64)
    y_test_enc  = np.array([label_to_index[v] for v in y_test], dtype=np.int64)

    class_names = [str(int(x)) for x in unique_y]

    print("\nLabel remapping:")
    for raw, enc in label_to_index.items():
        print(f"{raw} -> {enc}")

num_classes = len(class_names)
encoded_labels = np.arange(num_classes, dtype=np.int64)

print("\nNumber of classes :", num_classes)
print("Class names :", class_names)


# ============================================================
# ONE-HOT ENCODING FOR ELM
# ============================================================
def to_one_hot(y, num_classes):
    y = np.asarray(y, dtype=np.int64)
    if np.any(y < 0) or np.any(y >= num_classes):
        raise ValueError("Labels must be in [0, num_classes-1] for one-hot encoding.")
    return np.eye(num_classes, dtype=np.float32)[y]


T_train = to_one_hot(y_train_enc, num_classes)
T_val   = to_one_hot(y_val_enc, num_classes)
T_test  = to_one_hot(y_test_enc, num_classes)


# ============================================================
# ACTIVATION FUNCTION
# ============================================================
def sigmoid(z):
    z = np.clip(z, -50, 50)
    return 1.0 / (1.0 + np.exp(-z))


# ============================================================
# ELM TRAINING
# ============================================================
def train_elm(X, y_onehot, num_hidden=50, seed=10, reg=1e-6):
    """
    Multiclass ELM:
      - random hidden weights
      - analytical output weights with ridge regularization
    """
    rng = np.random.default_rng(seed)

    n_features = X.shape[1]

    W = rng.uniform(-1.0, 1.0, size=(num_hidden, n_features)).astype(np.float32)
    b = rng.uniform(0.0, 1.0, size=(num_hidden, 1)).astype(np.float32)

    H = sigmoid(W @ X.T + b).astype(np.float64)

    I = np.eye(num_hidden, dtype=np.float64)
    Beta = np.linalg.solve(H @ H.T + reg * I, H @ y_onehot.astype(np.float64))

    model = {
        "W": W,
        "b": b,
        "Beta": Beta,
        "num_hidden": num_hidden,
        "seed": seed,
        "reg": reg
    }
    return model


# ============================================================
# PREDICTION
# ============================================================
def predict_elm(model, X):
    W = model["W"]
    b = model["b"]
    Beta = model["Beta"]

    H = sigmoid(W @ X.T + b)
    scores = H.T @ Beta
    y_pred = np.argmax(scores, axis=1).astype(np.int64)

    return y_pred, scores


# ============================================================
# SEARCH FOR BEST NUMBER OF HIDDEN NEURONS
# ============================================================
hidden_candidates = [10, 20, 30, 50, 80, 100, 150, 200]

best_model = None
best_hidden = None
best_val_acc = -1.0

print("\nSearching for the best number of hidden neurons...")

for h in hidden_candidates:
    model = train_elm(X_train, T_train, num_hidden=h, seed=10, reg=1e-6)

    y_train_pred, _ = predict_elm(model, X_train)
    y_val_pred, _   = predict_elm(model, X_val)

    train_acc = accuracy_score(y_train_enc, y_train_pred) * 100
    val_acc   = accuracy_score(y_val_enc, y_val_pred) * 100

    print(f"Hidden={h:3d} | Train={train_acc:6.2f}% | Val={val_acc:6.2f}%")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_hidden = h
        best_model = model

print(f"\nBest hidden neuron count : {best_hidden}")
print(f"Validation accuracy      : {best_val_acc:.2f}%")


# ============================================================
# RETRAIN ON TRAIN + VALIDATION
# ============================================================
X_train_full = np.vstack([X_train, X_val]).astype(np.float32)
y_train_full = np.concatenate([y_train_enc, y_val_enc]).astype(np.int64)
T_train_full = to_one_hot(y_train_full, num_classes)

final_model = train_elm(
    X_train_full,
    T_train_full,
    num_hidden=best_hidden,
    seed=10,
    reg=1e-6
)


# ============================================================
# FINAL EVALUATION
# ============================================================
y_train_full_pred, _ = predict_elm(final_model, X_train_full)
y_test_pred, y_test_scores = predict_elm(final_model, X_test)

train_full_acc = accuracy_score(y_train_full, y_train_full_pred) * 100
test_acc = accuracy_score(y_test_enc, y_test_pred) * 100

print("\n================ FINAL RESULTS ================")
print(f"Train+Val accuracy : {train_full_acc:.2f} %")
print(f"Test accuracy      : {test_acc:.2f} %")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================
print("\nClassification report (test):")
print(classification_report(
    y_test_enc,
    y_test_pred,
    labels=encoded_labels,
    target_names=class_names,
    digits=4,
    zero_division=0
))


# ============================================================
# CONFUSION MATRIX
# ============================================================
fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay.from_predictions(
    y_test_enc,
    y_test_pred,
    labels=encoded_labels,
    display_labels=class_names,
    xticks_rotation=45,
    ax=ax,
    cmap="Blues"
)
plt.title(f"ELM Confusion Matrix (hidden={best_hidden})")
plt.tight_layout()
plt.show()


# ============================================================
# SIMPLE FEATURE IMPORTANCE INDICATOR
# ============================================================
feature_strength = np.mean(np.abs(final_model["W"]), axis=0)
order = np.argsort(feature_strength)[::-1]

print("\nTop features (simple indicator based on mean |W|):")
for idx in order[:15]:
    print(f"{feature_names[idx]} : {feature_strength[idx]:.6f}")

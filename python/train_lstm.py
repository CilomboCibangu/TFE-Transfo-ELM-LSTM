import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam


# ============================================================
# REPRODUCIBILITY
# ============================================================
np.random.seed(42)
tf.random.set_seed(42)
tf.keras.backend.clear_session()

print("TensorFlow version:", tf.__version__)


# ============================================================
# GPU
# ============================================================
gpus = tf.config.list_physical_devices('GPU')
has_gpu = len(gpus) > 0
print("GPU detected:", has_gpu)
if has_gpu:
    print("GPU:", gpus[0])


# ============================================================
# LOAD NPZ DATASET
# ============================================================
DATA_DIR = Path("DATASET_TRANSFO_TEST")
NPZ_PATH = DATA_DIR / "lstm_raw_cycle_light.npz"

if not NPZ_PATH.exists():
    raise FileNotFoundError(
        f"Dataset file not found: {NPZ_PATH}. "
        "Run preprocessing_lstm.py first."
    )

print("NPZ found:", NPZ_PATH)

data = np.load(NPZ_PATH, allow_pickle=True)

X_train = data["X_train"].astype(np.float32)
y_train = data["y_train"].astype(np.int64)

X_val = data["X_val"].astype(np.float32)
y_val = data["y_val"].astype(np.int64)

X_test = data["X_test"].astype(np.float32)
y_test = data["y_test"].astype(np.int64)

channel_names = data["channel_names"]
labels_encoded = data["labels_encoded"].astype(np.int64)
labels_original = data["labels_original"].astype(np.int64)

class_names = [f"CL{int(x)}" for x in labels_original]
num_classes = len(labels_encoded)

print("\nShapes:")
print("X_train :", X_train.shape, X_train.dtype)
print("X_val   :", X_val.shape, X_val.dtype)
print("X_test  :", X_test.shape, X_test.dtype)
print("Channels:", channel_names)
print("Classes :", class_names)


# ============================================================
# TRAINING PARAMETERS
# ============================================================
batch_size = 256
epochs = 20

print("\nTrain samples :", len(y_train))
print("Val samples   :", len(y_val))
print("Test samples  :", len(y_test))
print("Batch size    :", batch_size)
print("Steps/epoch train ≈", int(np.ceil(len(y_train) / batch_size)))
print("Steps/epoch val   ≈", int(np.ceil(len(y_val) / batch_size)))
print("Steps/epoch test  ≈", int(np.ceil(len(y_test) / batch_size)))


# ============================================================
# TF.DATA PIPELINES
# ============================================================
AUTO = tf.data.AUTOTUNE

train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_ds = train_ds.shuffle(
    buffer_size=min(len(y_train), 200_000),
    seed=42,
    reshuffle_each_iteration=True
).batch(batch_size, drop_remainder=False).prefetch(AUTO)

val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
val_ds = val_ds.batch(batch_size, drop_remainder=False).prefetch(AUTO)

test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
test_ds = test_ds.batch(batch_size, drop_remainder=False).prefetch(AUTO)


# ============================================================
# MODEL
# ============================================================
model = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])),

    LSTM(48, return_sequences=True),
    Dropout(0.20),

    LSTM(24, return_sequences=False),
    Dropout(0.20),

    Dense(24, activation="relu"),
    Dense(num_classes, activation="softmax")
])


# ============================================================
# COMPILE
# ============================================================
model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
    steps_per_execution=64
)

model.summary()


# ============================================================
# CALLBACKS
# ============================================================
callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=1,
        min_lr=1e-5,
        verbose=1
    )
]


# ============================================================
# TRAINING
# ============================================================
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=callbacks,
    validation_freq=2,
    verbose=2
)


# ============================================================
# EVALUATION
# ============================================================
test_loss, test_acc = model.evaluate(test_ds, verbose=0)
print(f"\nTest accuracy : {100 * test_acc:.2f} %")
print(f"Test loss     : {test_loss:.4f}")

y_prob = model.predict(test_ds, verbose=0)
y_pred = np.argmax(y_prob, axis=1).astype(np.int64)

print("\nClassification report (test):")
print(classification_report(
    y_test,
    y_pred,
    labels=labels_encoded,
    target_names=class_names,
    digits=4,
    zero_division=0
))

cm = confusion_matrix(y_test, y_pred, labels=labels_encoded)

fig, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(ax=ax, cmap="Blues", colorbar=True)
plt.title("LSTM raw cycle - Confusion Matrix")
plt.tight_layout()
plt.show()


# ============================================================
# LEARNING CURVES
# ============================================================
train_epochs = np.arange(1, len(history.history["loss"]) + 1)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_epochs, history.history["loss"], label="train_loss")
if "val_loss" in history.history and len(history.history["val_loss"]) > 0:
    val_epochs = np.arange(2, 2 * len(history.history["val_loss"]) + 1, 2)
    plt.plot(val_epochs, history.history["val_loss"], label="val_loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(train_epochs, history.history["accuracy"], label="train_acc")
if "val_accuracy" in history.history and len(history.history["val_accuracy"]) > 0:
    val_epochs = np.arange(2, 2 * len(history.history["val_accuracy"]) + 1, 2)
    plt.plot(val_epochs, history.history["val_accuracy"], label="val_acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy Curve")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# ============================================================
# SAVE MODEL
# ============================================================
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

model.save(MODELS_DIR / "best_lstm.keras")
print("\nSaved model:", MODELS_DIR / "best_lstm.keras")

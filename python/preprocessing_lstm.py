import re
import numpy as np
import pandas as pd

from pathlib import Path


# ============================================================
# PATHS
# ============================================================
DATA_DIR = Path("DATASET_TRANSFO_TEST")

COMMON_SPLIT_FILE = DATA_DIR / "common_file_split.csv"
INDEX_FILE = DATA_DIR / "index.csv"
OUT_NPZ = DATA_DIR / "lstm_raw_cycle_light.npz"

print("DATA_DIR =", DATA_DIR)
print("Exists ?", DATA_DIR.exists())
print("Split file =", COMMON_SPLIT_FILE, "| Exists ?", COMMON_SPLIT_FILE.exists())


# ============================================================
# PARAMETERS
# ============================================================
GRID_FREQ_HZ = 50.0
WINDOW_CYCLES = 1.0
STEP_CYCLES = 1.0

WINDOW_SEC = WINDOW_CYCLES / GRID_FREQ_HZ
STEP_SEC = STEP_CYCLES / GRID_FREQ_HZ

MIN_SAMPLES_PER_WINDOW = 8
SEQ_LEN = 16
STORE_DTYPE = np.float16
N_CHANNELS_EXPECTED = 8

CHANNELS = [
    "I_MTa", "I_MTb", "I_MTc",
    "T_oil", "T_wdg_A", "T_wdg_B", "T_wdg_C", "T_amb"
]

if not COMMON_SPLIT_FILE.exists():
    raise FileNotFoundError(
        "common_file_split.csv not found. It must be created first using the common split procedure."
    )


# ============================================================
# UTILITIES
# ============================================================
def portable_basename(path_like: str) -> str:
    return re.split(r"[\\/]", str(path_like).strip())[-1]


def read_csv_auto(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=';')
    df.columns = df.columns.str.strip()
    return df


def build_meta_map(index_file: Path) -> dict:
    meta_map = {}
    if not index_file.exists():
        return meta_map

    index_df = pd.read_csv(index_file, sep=';')
    index_df.columns = index_df.columns.str.strip()

    if "csv" not in index_df.columns:
        return meta_map

    for _, r in index_df.iterrows():
        fname = portable_basename(str(r["csv"]))
        meta_map[fname] = r.to_dict()

    return meta_map


def ensure_required_channels(df: pd.DataFrame, meta_row: dict | None = None) -> pd.DataFrame:
    df = df.copy()

    if "Temps" not in df.columns:
        raise ValueError("Missing 'Temps' column.")

    if "T_amb" not in df.columns:
        if meta_row is not None and "Tamb" in meta_row and not pd.isna(meta_row["Tamb"]):
            df["T_amb"] = float(meta_row["Tamb"])
        else:
            raise ValueError("Missing 'T_amb' column and Tamb unavailable in index.csv.")

    required = ["Temps"] + CHANNELS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    return df


def clean_time_axis(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values("Temps").reset_index(drop=True)
    df = df.drop_duplicates(subset=["Temps"], keep="first").reset_index(drop=True)

    if len(df) < 2:
        raise ValueError("Time series too short after cleaning.")

    return df


def resample_window_to_sequence(w: pd.DataFrame, channels, seq_len: int) -> np.ndarray:
    t = w["Temps"].to_numpy(dtype=float)
    t = t - t[0]

    if len(t) < 2 or np.allclose(t[-1], 0.0):
        raise ValueError("Invalid or too short time window.")

    t_new = np.linspace(t[0], t[-1], seq_len)
    arr = np.zeros((seq_len, len(channels)), dtype=np.float32)

    for j, col in enumerate(channels):
        y = w[col].to_numpy(dtype=float)
        arr[:, j] = np.interp(t_new, t, y).astype(np.float32)

    return arr


def iter_sequences_from_file(csv_path: Path, meta_row: dict | None):
    df = read_csv_auto(csv_path)
    df = ensure_required_channels(df, meta_row=meta_row)
    df = clean_time_axis(df)

    t = df["Temps"].to_numpy(dtype=float)
    t0 = float(t[0])
    t1 = float(t[-1])

    start = t0

    while start + WINDOW_SEC <= t1 + 1e-12:
        stop = start + WINDOW_SEC
        w = df.loc[(df["Temps"] >= start) & (df["Temps"] < stop)].copy()

        if len(w) >= MIN_SAMPLES_PER_WINDOW:
            yield resample_window_to_sequence(w, CHANNELS, SEQ_LEN)

        start += STEP_SEC


# ============================================================
# LOAD COMMON SPLIT
# ============================================================
split_df = pd.read_csv(COMMON_SPLIT_FILE, sep=';')
split_df.columns = split_df.columns.str.strip()

required_cols = {"file_name", "csv_path", "label_raw", "label", "split"}
missing = required_cols - set(split_df.columns)
if missing:
    raise ValueError(f"Missing columns in common_file_split.csv: {missing}")

split_df["file_name"] = split_df["file_name"].astype(str)
split_df["csv_path"] = split_df["csv_path"].astype(str)
split_df["label_raw"] = split_df["label_raw"].astype(np.int64)
split_df["label"] = split_df["label"].astype(np.int64)
split_df["split"] = split_df["split"].astype(str)

meta_map = build_meta_map(INDEX_FILE)

print("\nFile distribution by split:")
print(split_df["split"].value_counts())

print("\nRaw labels:")
print(np.sort(split_df["label_raw"].unique()))


# ============================================================
# PASS 1: COMPUTE NORMALIZATION ON TRAIN ONLY
# ============================================================
sum_ch = np.zeros(len(CHANNELS), dtype=np.float64)
sqsum_ch = np.zeros(len(CHANNELS), dtype=np.float64)
n_points_total = 0
n_windows_train = 0

train_files = split_df[split_df["split"] == "train"].reset_index(drop=True)

for i, r in train_files.iterrows():
    csv_path = Path(r["csv_path"])
    file_name = r["file_name"]
    meta_row = meta_map.get(file_name, {})

    for seq in iter_sequences_from_file(csv_path, meta_row):
        sum_ch += seq.sum(axis=0, dtype=np.float64)
        sqsum_ch += np.square(seq, dtype=np.float64).sum(axis=0, dtype=np.float64)
        n_points_total += seq.shape[0]
        n_windows_train += 1

    if (i + 1) % 20 == 0 or (i + 1) == len(train_files):
        print(f"Pass 1 - train stats: {i+1}/{len(train_files)} files")

if n_points_total == 0:
    raise ValueError("No valid train windows found to compute mean/std.")

mean_ch_1d = sum_ch / n_points_total
var_ch_1d = sqsum_ch / n_points_total - np.square(mean_ch_1d)
var_ch_1d = np.maximum(var_ch_1d, 1e-12)
std_ch_1d = np.sqrt(var_ch_1d)

mean_ch = mean_ch_1d.reshape(1, 1, -1).astype(np.float32)
std_ch = std_ch_1d.reshape(1, 1, -1).astype(np.float32)

print("\nNormalization computed.")
print("Number of train windows:", n_windows_train)


# ============================================================
# PASS 2: COUNT WINDOWS PER SPLIT
# ============================================================
counts = {"train": 0, "val": 0, "test": 0}

for i, r in split_df.iterrows():
    csv_path = Path(r["csv_path"])
    file_name = r["file_name"]
    split_name = r["split"]
    meta_row = meta_map.get(file_name, {})

    c = 0
    for _ in iter_sequences_from_file(csv_path, meta_row):
        c += 1

    counts[split_name] += c

    if (i + 1) % 20 == 0 or (i + 1) == len(split_df):
        print(f"Pass 2 - counting: {i+1}/{len(split_df)} files")

print("\nWindow count by split:")
print(counts)

if counts["train"] == 0 or counts["val"] == 0 or counts["test"] == 0:
    raise ValueError("One of the splits is empty.")


# ============================================================
# DIRECT ALLOCATION
# ============================================================
X_train = np.empty((counts["train"], SEQ_LEN, len(CHANNELS)), dtype=STORE_DTYPE)
y_train = np.empty((counts["train"],), dtype=np.int64)

X_val = np.empty((counts["val"], SEQ_LEN, len(CHANNELS)), dtype=STORE_DTYPE)
y_val = np.empty((counts["val"],), dtype=np.int64)

X_test = np.empty((counts["test"], SEQ_LEN, len(CHANNELS)), dtype=STORE_DTYPE)
y_test = np.empty((counts["test"],), dtype=np.int64)

ptr = {"train": 0, "val": 0, "test": 0}


# ============================================================
# PASS 3: FILL ARRAYS
# ============================================================
for i, r in split_df.iterrows():
    csv_path = Path(r["csv_path"])
    file_name = r["file_name"]
    label = int(r["label"])
    split_name = r["split"]
    meta_row = meta_map.get(file_name, {})

    for seq in iter_sequences_from_file(csv_path, meta_row):
        seq_norm = ((seq - mean_ch_1d) / std_ch_1d).astype(STORE_DTYPE)

        k = ptr[split_name]

        if split_name == "train":
            X_train[k] = seq_norm
            y_train[k] = label
        elif split_name == "val":
            X_val[k] = seq_norm
            y_val[k] = label
        else:
            X_test[k] = seq_norm
            y_test[k] = label

        ptr[split_name] += 1

    if (i + 1) % 20 == 0 or (i + 1) == len(split_df):
        print(f"Pass 3 - filling arrays: {i+1}/{len(split_df)} files")

labels_original = np.sort(split_df["label_raw"].unique()).astype(np.int64)
labels_encoded = np.sort(split_df["label"].unique()).astype(np.int64)

print("\nFinal shapes:")
print("X_train :", X_train.shape, y_train.shape, X_train.dtype)
print("X_val   :", X_val.shape, y_val.shape, X_val.dtype)
print("X_test  :", X_test.shape, y_test.shape, X_test.dtype)


# ============================================================
# SAVE LIGHTWEIGHT NPZ
# ============================================================
np.savez_compressed(
    OUT_NPZ,
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    X_test=X_test,
    y_test=y_test,
    mean_ch=mean_ch,
    std_ch=std_ch,
    channel_names=np.array(CHANNELS),
    seq_len=np.array([SEQ_LEN], dtype=np.int64),
    window_sec=np.array([WINDOW_SEC], dtype=np.float32),
    grid_freq_hz=np.array([GRID_FREQ_HZ], dtype=np.float32),
    labels_encoded=labels_encoded,
    labels_original=labels_original
)

print("\nSaved file:")
print(OUT_NPZ)

import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# PARAMETERS
# ============================================================
DATA_DIR = Path("DATASET_TRANSFO_TEST")
INDEX_FILE = DATA_DIR / "index.csv"

OUT_NPZ = DATA_DIR / "elm_dataset_ready.npz"
OUT_FEATURES_CSV = DATA_DIR / "elm_features_table.csv"

# Grid frequency
GRID_FREQ_HZ = 50.0   # set 60.0 if using a 60 Hz grid

# Windowing
WINDOW_CYCLES = 1.0   # 1 cycle
STEP_CYCLES = 1.0     # no overlap

WINDOW_SEC = WINDOW_CYCLES / GRID_FREQ_HZ
STEP_SEC = STEP_CYCLES / GRID_FREQ_HZ

# Minimum number of samples required to keep a window
MIN_SAMPLES_PER_WINDOW = 8

# Dataset split
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_STATE = 42


# ============================================================
# UTILITIES
# ============================================================
def read_csv_auto(path: Path) -> pd.DataFrame:
    """Read CSV using ';' as separator."""
    return pd.read_csv(path, sep=';')


def infer_label_from_name(name: str) -> int:
    """
    Infer class label from filename.
    Example:
      sim_CL6_PUMPFAIL_...
      sim_TEST_CL5_FAN_FAIL.csv
    """
    m = re.search(r'CL(\d+)', name, flags=re.IGNORECASE)
    if m is None:
        raise ValueError(f"Unable to infer label from filename: {name}")
    return int(m.group(1))


def resolve_csv_path(csv_value: str, data_dir: Path) -> Path:
    """
    If index.csv contains an absolute path, use it.
    Otherwise, search using the basename inside DATA_DIR.
    """
    p = Path(str(csv_value))
    if p.exists():
        return p

    p2 = data_dir / p.name
    if p2.exists():
        return p2

    raise FileNotFoundError(f"CSV file not found: {csv_value}")


def rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(np.square(x))))


def extract_features_from_window(w: pd.DataFrame, meta_row: Optional[Dict] = None) -> Dict:
    """
    Extract one feature row from a single time window corresponding to 1 grid cycle.
    """
    t = w["Temps"].to_numpy(dtype=float)

    ia = w["I_MTa"].to_numpy(dtype=float)
    ib = w["I_MTb"].to_numpy(dtype=float)
    ic = w["I_MTc"].to_numpy(dtype=float)

    toil = w["T_oil"].to_numpy(dtype=float)
    twa = w["T_wdg_A"].to_numpy(dtype=float)
    twb = w["T_wdg_B"].to_numpy(dtype=float)
    twc = w["T_wdg_C"].to_numpy(dtype=float)

    if "T_amb" in w.columns:
        tamb_mean = float(np.mean(w["T_amb"].to_numpy(dtype=float)))
    else:
        if meta_row is not None and "Tamb" in meta_row:
            tamb_mean = float(meta_row["Tamb"])
        else:
            tamb_mean = np.nan

    # Current RMS features
    Irms_A = rms(ia)
    Irms_B = rms(ib)
    Irms_C = rms(ic)

    I_mean = float(np.mean([Irms_A, Irms_B, Irms_C]))
    I_std = float(np.std([Irms_A, Irms_B, Irms_C], ddof=0))
    I_unb = float(np.max([Irms_A, Irms_B, Irms_C]) - np.min([Irms_A, Irms_B, Irms_C]))

    eps = 1e-12
    I_A_ratio = float(Irms_A / (I_mean + eps))
    I_B_ratio = float(Irms_B / (I_mean + eps))
    I_C_ratio = float(Irms_C / (I_mean + eps))

    # Thermal features
    Toil_mean = float(np.mean(toil))
    TwA_mean = float(np.mean(twa))
    TwB_mean = float(np.mean(twb))
    TwC_mean = float(np.mean(twc))

    Tw_max = float(np.max([TwA_mean, TwB_mean, TwC_mean]))
    Tw_min = float(np.min([TwA_mean, TwB_mean, TwC_mean]))
    Tw_spread = float(Tw_max - Tw_min)

    DeltaA = float(TwA_mean - Toil_mean)
    DeltaB = float(TwB_mean - Toil_mean)
    DeltaC = float(TwC_mean - Toil_mean)
    DeltaMax = float(Tw_max - Toil_mean)

    feat = {
        "Irms_A": Irms_A,
        "Irms_B": Irms_B,
        "Irms_C": Irms_C,
        "I_mean": I_mean,
        "I_std": I_std,
        "I_unb": I_unb,
        "I_A_ratio": I_A_ratio,
        "I_B_ratio": I_B_ratio,
        "I_C_ratio": I_C_ratio,
        "Toil_mean": Toil_mean,
        "TwA_mean": TwA_mean,
        "TwB_mean": TwB_mean,
        "TwC_mean": TwC_mean,
        "Tw_max": Tw_max,
        "DeltaA": DeltaA,
        "DeltaB": DeltaB,
        "DeltaC": DeltaC,
        "DeltaMax": DeltaMax,
        "Tw_spread": Tw_spread,
        "Tamb_mean": tamb_mean,
        "samples_in_window": int(len(w)),
        "t_start_real": float(t[0]),
        "t_end_real": float(t[-1]),
    }

    return feat


def extract_feature_rows_from_csv(
    csv_path: Path,
    label: int,
    meta_row: Optional[Dict] = None
) -> List[Dict]:
    """
    Process a full simulation file and generate one feature row per cycle.
    """
    df = read_csv_auto(csv_path)

    required = [
        "Temps",
        "I_MTa", "I_MTb", "I_MTc",
        "T_oil", "T_wdg_A", "T_wdg_B", "T_wdg_C"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path.name}: missing columns {missing}")

    df = df.sort_values("Temps").reset_index(drop=True)

    t = df["Temps"].to_numpy(dtype=float)
    t0 = float(t[0])
    t1 = float(t[-1])

    records: List[Dict] = []
    start = t0
    cycle_id = 0

    while start + WINDOW_SEC <= t1 + 1e-12:
        stop = start + WINDOW_SEC
        w = df.loc[(df["Temps"] >= start) & (df["Temps"] < stop)].copy()

        if len(w) >= MIN_SAMPLES_PER_WINDOW:
            feat = extract_features_from_window(w, meta_row=meta_row)
            feat["label"] = int(label)
            feat["csv_path"] = str(csv_path)
            feat["cycle_id"] = int(cycle_id)
            feat["window_t0"] = float(start)
            feat["window_t1"] = float(stop)

            for k in ["Mode", "Fault", "Tamb", "cosphi", "Kpump", "Kfan", "PhaseMono", "unbal_pct"]:
                if isinstance(meta_row, dict) and k in meta_row:
                    feat[f"meta_{k}"] = meta_row[k]

            records.append(feat)

        start += STEP_SEC
        cycle_id += 1

    return records


def build_records_for_split(rows: List[Dict], file_indices_subset: np.ndarray, split_name: str) -> List[Dict]:
    """Extract all windows from all files belonging to one split."""
    out: List[Dict] = []

    for idx in file_indices_subset:
        item = rows[int(idx)]
        csv_path = Path(item["csv_path"])
        label = int(item["label"])
        meta = item["meta"]

        recs = extract_feature_rows_from_csv(csv_path, label=label, meta_row=meta)

        for r in recs:
            r["split"] = split_name

        out.extend(recs)

    return out


# ============================================================
# LOAD FILES
# ============================================================
rows: List[Dict] = []

if INDEX_FILE.exists():
    index_df = pd.read_csv(INDEX_FILE, sep=';')

    if "csv" not in index_df.columns:
        raise ValueError("index.csv exists but the 'csv' column is missing.")

    label_col = None
    for cand in ["Label", "label"]:
        if cand in index_df.columns:
            label_col = cand
            break

    for _, r in index_df.iterrows():
        csv_path = resolve_csv_path(str(r["csv"]), DATA_DIR)

        if label_col is not None:
            label = int(r[label_col])
        else:
            label = infer_label_from_name(csv_path.name)

        rows.append({
            "csv_path": str(csv_path),
            "label": label,
            "meta": r.to_dict()
        })

else:
    csv_files = sorted(DATA_DIR.glob("sim_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No sim_*.csv files found in {DATA_DIR}")

    for p in csv_files:
        label = infer_label_from_name(p.name)
        rows.append({
            "csv_path": str(p),
            "label": label,
            "meta": {}
        })

print(f"{len(rows)} files found.")


# ============================================================
# STRATIFIED SPLIT BY FILE
# IMPORTANT: split before feature extraction
# ============================================================
file_labels = np.array([int(r["label"]) for r in rows], dtype=np.int64)
file_indices = np.arange(len(rows))

n_classes = len(np.unique(file_labels))
min_count = pd.Series(file_labels).value_counts().min()

if n_classes < 2 or min_count < 3:
    raise ValueError(
        "Not enough classes or examples per class for a proper stratified split."
    )

idx_train_val_files, idx_test_files = train_test_split(
    file_indices,
    test_size=TEST_RATIO,
    stratify=file_labels,
    random_state=RANDOM_STATE
)

y_train_val_files = file_labels[idx_train_val_files]
val_relative = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)

idx_train_files, idx_val_files = train_test_split(
    idx_train_val_files,
    test_size=val_relative,
    stratify=y_train_val_files,
    random_state=RANDOM_STATE
)

print(f"Train files: {len(idx_train_files)}")
print(f"Validation files: {len(idx_val_files)}")
print(f"Test files: {len(idx_test_files)}")


# ============================================================
# FEATURE EXTRACTION BY SPLIT
# ============================================================
records_train = build_records_for_split(rows, idx_train_files, "train")
print(f"Extracted train cycles: {len(records_train)}")

records_val = build_records_for_split(rows, idx_val_files, "val")
print(f"Extracted validation cycles: {len(records_val)}")

records_test = build_records_for_split(rows, idx_test_files, "test")
print(f"Extracted test cycles: {len(records_test)}")

records = records_train + records_val + records_test

if len(records) == 0:
    raise ValueError("No valid windows extracted. Check GRID_FREQ_HZ or MIN_SAMPLES_PER_WINDOW.")

features_df = pd.DataFrame(records)


# ============================================================
# FEATURES USED FOR TRAINING
# ============================================================
feature_names = [
    "Irms_A", "Irms_B", "Irms_C",
    "I_mean", "I_std", "I_unb",
    "I_A_ratio", "I_B_ratio", "I_C_ratio",
    "Toil_mean", "TwA_mean", "TwB_mean", "TwC_mean",
    "Tw_max",
    "DeltaA", "DeltaB", "DeltaC", "DeltaMax",
    "Tw_spread",
    "Tamb_mean"
]

for c in feature_names + ["label", "split", "csv_path"]:
    if c not in features_df.columns:
        raise ValueError(f"Expected missing column: {c}")


# ============================================================
# CLEANING
# ============================================================
before = len(features_df)

features_df = (
    features_df
    .replace([np.inf, -np.inf], np.nan)
    .dropna(subset=feature_names + ["label", "split"])
    .reset_index(drop=True)
)

after = len(features_df)

print(f"Rows before cleaning: {before}")
print(f"Rows after cleaning: {after}")

print("\nClass distribution:")
print(features_df["label"].value_counts().sort_index())

print("\nSplit distribution:")
print(features_df["split"].value_counts())


# ============================================================
# SPLIT INDICES AFTER CLEANING
# ============================================================
idx_train = np.where(features_df["split"].to_numpy() == "train")[0]
idx_val = np.where(features_df["split"].to_numpy() == "val")[0]
idx_test = np.where(features_df["split"].to_numpy() == "test")[0]

if len(idx_train) == 0 or len(idx_val) == 0 or len(idx_test) == 0:
    raise ValueError("One of the splits is empty after cleaning.")


# ============================================================
# MATRICES X / y
# ============================================================
X = features_df[feature_names].to_numpy(dtype=np.float32)
y = features_df["label"].to_numpy(dtype=np.int64)

X_train_raw = X[idx_train]
y_train = y[idx_train]

X_val_raw = X[idx_val]
y_val = y[idx_val]

X_test_raw = X[idx_test]
y_test = y[idx_test]


# ============================================================
# Z-SCORE NORMALIZATION
# IMPORTANT: computed on training set only
# ============================================================
mu = X_train_raw.mean(axis=0)
sigma = X_train_raw.std(axis=0)
sigma[sigma < 1e-12] = 1.0

X_train = ((X_train_raw - mu) / sigma).astype(np.float32)
X_val = ((X_val_raw - mu) / sigma).astype(np.float32)
X_test = ((X_test_raw - mu) / sigma).astype(np.float32)
X_all_z = ((X - mu) / sigma).astype(np.float32)


# ============================================================
# SAVE OUTPUTS
# ============================================================
features_df.to_csv(OUT_FEATURES_CSV, sep=';', index=False)

np.savez_compressed(
    OUT_NPZ,
    X_raw=X.astype(np.float32),
    y=y.astype(np.int64),

    X_all_z=X_all_z.astype(np.float32),

    X_train=X_train,
    y_train=y_train.astype(np.int64),

    X_val=X_val,
    y_val=y_val.astype(np.int64),

    X_test=X_test,
    y_test=y_test.astype(np.int64),

    idx_train=idx_train.astype(np.int64),
    idx_val=idx_val.astype(np.int64),
    idx_test=idx_test.astype(np.int64),

    mean=mu.astype(np.float32),
    std=sigma.astype(np.float32),

    feature_names=np.array(feature_names),
    csv_paths=features_df["csv_path"].astype(str).to_numpy(),
    splits=features_df["split"].astype(str).to_numpy(),

    labels=np.sort(features_df["label"].unique()).astype(np.int64)
)

print("\nDone.")
print(f"Saved NPZ file: {OUT_NPZ}")
print(f"Saved feature table: {OUT_FEATURES_CSV}")
print(f"Total X shape: {X.shape}")
print(f"Train / Val / Test: {X_train.shape[0]} / {X_val.shape[0]} / {X_test.shape[0]}")

import numpy as np
import pandas as pd
import os
import random

BIS_events = pd.read_csv("../data/derivatives/meta/BIS_events.csv")
high_events = BIS_events.query("state == 'high' and duration > 60")
low_events = BIS_events.query("state == 'low'")
epoch_save_dir = "../data/derivatives/HR_epochs/BIS_prediction/"
os.makedirs(epoch_save_dir, exist_ok=True)


def remove_close_events(df, time_threshold=1200):
    keep = []
    dropped = []
    last_onset = -float("inf")

    for idx, row in df.iterrows():
        if row["onset"] - last_onset >= time_threshold:
            keep.append(True)
            last_onset = row["onset"]
        else:
            keep.append(False)
            dropped.append(row)
    if dropped:
        print(f"Dropped events for subject {df.iloc[0]['subject_id']}:")
        for r in dropped:
            print(f"  onset = {r['onset']}")

    return df[keep]


high_events = high_events.sort_values(["subject_id", "onset"])
high_events = high_events.groupby("subject_id", group_keys=False).apply(
    remove_close_events
)

# Extracting positive epochs
HR_fs = 10
WINDOW = 300 * HR_fs  # 300 seconds window
for i, row in high_events.iterrows():
    BIS = np.load(row["BIS_path"])
    HR = np.load(row["HR_path"])
    for pre_high in [0, 5, 10, 15]:
        start_time = row["onset"] - (pre_high + 5) * 60
        duration = 300
        subject_id = row["subject_id"]
        state = f"Pre-{pre_high}min"

        start_point = int(start_time * HR_fs)
        end_point = int((start_time + duration) * HR_fs)
        epoch = HR[start_point : (start_point + WINDOW)]

        BIS_epoch = BIS[start_point // 10 : (start_point + WINDOW) // 10]
        print(np.max(BIS_epoch), np.min(BIS_epoch), np.mean(BIS_epoch))
        epoch_filename = f"{epoch_save_dir}/{subject_id}_{state}_{start_point}_{start_point + WINDOW}.npy"
        np.save(epoch_filename, epoch)

epoch_save_dir = "../data/derivatives/HR_epochs/BIS_prediction/"
os.makedirs(epoch_save_dir, exist_ok=True)

# Extracting negative epochs
min_gap_sec = 1200
min_gap_pts = min_gap_sec * HR_fs

for i, row in low_events.iterrows():
    HR = np.load(row["HR_path"])
    start_time = row["onset"]
    duration = row["duration"]
    subject_id = row["subject_id"]
    if subject_id not in high_events.subject_id.values:
        continue
    state = row["state"]

    start_point = int(start_time * HR_fs)
    end_point = int((start_time + duration) * HR_fs)

    seg_max_end = end_point - min_gap_pts
    max_start = seg_max_end - WINDOW

    if max_start <= start_point:
        print(f"Not enough safe window for {subject_id} at {start_time}s, skipping.")
        continue

    n_epochs = random.randint(1, 2)
    used_intervals = []

    for _ in range(n_epochs):
        attempt = 0
        while attempt < 10:
            rand_start = random.randint(start_point, max_start)
            rand_end = rand_start + WINDOW

            if all(abs(rand_start - s) >= WINDOW for s in used_intervals):
                used_intervals.append(rand_start)
                epoch = HR[rand_start:rand_end]

                epoch_filename = f"{epoch_save_dir}/{subject_id}_Negative_{rand_start}_{rand_end}.npy"
                np.save(epoch_filename, epoch)
                break
            else:
                attempt += 1
detection_epochs_paths = [
    epoch_save_dir + f for f in os.listdir(epoch_save_dir) if f.endswith(".npy")
]


def get_info(path):
    file_name = os.path.basename(path)
    subject_id, state, start_point, end_point = file_name.split("_")
    return subject_id, state, start_point, end_point[:-4], path


data_info = zip(*[get_info(path) for path in detection_epochs_paths])
subject_ids, states, start_points, end_points, paths = data_info
data_info = pd.DataFrame(
    {
        "subject_id": subject_ids,
        "state": states,
        "start_point": start_points,
        "end_point": end_points,
        "epoch_path": paths,
    }
)
data_info.to_csv("../data/derivatives/meta/HR_BIS_prediction_epochs.csv", index=False)
print(data_info["state"].value_counts())

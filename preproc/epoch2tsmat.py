import os
import pandas as pd
import numpy as np
from scipy.io import savemat

meta = pd.read_csv("../data/derivatives/meta/HR_BIS_prediction_epochs.csv")
subjects = meta["subject_id"].unique()
os.makedirs(
    "../data/derivatives/hctsa/BIS_prediction/tsmat",
    exist_ok=True,
)
for subject in subjects:
    save_path = f"../data/derivatives/hctsa/BIS_prediction/tsmat/{subject}.mat"
    subject_meta = meta[meta["subject_id"] == subject]
    subject_epochs = np.array([np.load(path) for path in subject_meta["epoch_path"]])
    labels = (
        subject_meta["subject_id"].astype(str)
        + "_"
        + subject_meta["state"]
        + "_"
        + subject_meta["start_point"].astype(str)
        + "_"
        + subject_meta["end_point"].astype(str)
    ).tolist()
    keywords = labels.copy()
    hctsa_data = {
        "timeSeriesData": subject_epochs,
        "labels": np.array(labels, dtype=object),
        "keywords": np.array(keywords, dtype=object),
    }
    savemat(save_path, hctsa_data)

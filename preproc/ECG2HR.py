import vitaldb
import pandas as pd
import numpy as np
import neurokit2 as nk
from pathlib import Path
import os
from tqdm import tqdm
from joblib import Parallel, delayed

vital_dir = Path("../data/vital_files/")
vital_paths = list(vital_dir.glob("*.vital"))


def HR_analyze(path):
    # Load the data
    vital = vitaldb.read_vital(str(path))
    ECG = vital.get_track_samples("SNUADC/ECG_II", 1 / 500)
    HR_full_len = np.empty(ECG.shape[0])
    HR = nk.ecg_process(ECG[~np.isnan(ECG)], 500)[0]["ECG_Rate"].values
    HR_full_len[~np.isnan(ECG)] = HR
    HR_full_len[np.isnan(ECG)] = np.nan
    return HR


def make_save_path(path):
    os.makedirs("../data/derivatives/", exist_ok=True)
    return "../data/derivatives/" + path.stem + "_HR.npy"


def process(path):
    try:
        HR = HR_analyze(path)
        save_path = make_save_path(path)
        np.save(save_path, HR)
        return None
    except:
        return path


results = Parallel(n_jobs=10)(delayed(process)(path) for path in tqdm(vital_paths))

failed_paths = [path for path in results if path is not None]

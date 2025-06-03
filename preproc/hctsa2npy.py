import matlab.engine
import numpy as np
import os

hctsa_mats_source = "../data/derivatives/hctsa/BIS_prediction/hctsa/"
hctsa_mats_files = [f for f in os.listdir(hctsa_mats_source) if f.endswith(".mat")]
hctsa_mats_paths = [hctsa_mats_source + f for f in hctsa_mats_files]
hctsa_np_target = "../data/derivatives/hctsa/BIS_prediction/hctsa_npy/"
if not os.path.exists(hctsa_np_target):
    os.makedirs(hctsa_np_target)
hctsa_np_paths = [hctsa_np_target + f.replace(".mat", ".npy") for f in hctsa_mats_files]
eng = matlab.engine.start_matlab()
for hctsa_mats_path, hctsa_np_path in zip(hctsa_mats_paths, hctsa_np_paths):
    print(f"Processing {hctsa_mats_path}")
    if os.path.exists(hctsa_np_path):
        print(f"Skipping {hctsa_np_path}, already transformed")
        continue
    data = eng.load(hctsa_mats_path)
    try:
        TS_DataMat = data["TS_DataMat"]
        TS_DataMat = np.array(TS_DataMat)
        eng.workspace["TimeSeries"] = data["TimeSeries"]
        TS_labels = eng.eval("TimeSeries.Name", nargout=1)
        np.save(hctsa_np_path, TS_DataMat)
        with open(hctsa_np_path.replace(".npy", "_label.txt"), "w") as f:
            for name in TS_labels:
                f.write(f"{name}\n")
    except:
        print(f"Error processing {hctsa_mats_path}")
        continue
eng.quit()

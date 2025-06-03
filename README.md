# HR2BIS

## Folder Structure

### `preproc/`
- `ECG2HR.py`: Converts raw ECG signals into heart rate (HR) time series.
- `HR2epoch.py`: Extracts HR epochs aligned with BIS events.
- `epoch2tsmat.py`: Converts HR epochs into the hctsa input format.
- `hctsa.m`: Performs time-series feature extraction using the hctsa framework.
- `hctsa2npy.py`: Converts hctsa output files into NumPy array format.

### `train/`
- `prediction.py`: Trains and evaluates LightGBM models using the full set of hctsa features.
- `compact_set_prediction.py`: Trains LightGBM models using various compact feature sets.
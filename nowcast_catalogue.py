import pandas as pd
import numpy as np
import os
from scipy.signal import find_peaks

def load_data(csv_path="raw_flux_data.csv"):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing {csv_path}. Run create_ml_dataset.py first.")
    
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df.index.name = 'time'
    return df

def classify_flare(flux_val):
    if flux_val >= 1e-4: return 'X-class'
    if flux_val >= 1e-5: return 'M-class'
    if flux_val >= 1e-6: return 'C-class'
    if flux_val >= 1e-7: return 'B-class'
    return 'A-class'

def detect_flares(series, threshold, prominence, distance, channel_name):
    """
    Detect peaks in a time series using scipy.signal.find_peaks.
    """
    # Find peaks
    peaks, properties = find_peaks(series, height=threshold, prominence=prominence, distance=distance)
    
    events = []
    for i, peak_idx in enumerate(peaks):
        peak_time = series.index[peak_idx]
        peak_flux = series.iloc[peak_idx]
        
        events.append({
            'channel': channel_name,
            'peak_time': peak_time,
            'peak_flux': peak_flux,
            'flare_class': classify_flare(peak_flux) if 'Soft' in channel_name else 'N/A'
        })
    return pd.DataFrame(events)

def main():
    print("Loading raw X-ray data for Nowcasting...")
    df = load_data()
    
    # Soft X-ray Nowcasting (SoLEXS proxy)
    # Detect C-class and above (> 1e-6)
    print("Running detection on Soft X-rays (SoLEXS proxy)...")
    soft_events = detect_flares(df['xrsb'], threshold=1e-6, prominence=5e-7, distance=60, channel_name='Soft (SoLEXS proxy)')
    
    # Hard X-ray Nowcasting (HEL1OS proxy)
    # Hard X-rays have lower flux, so lower threshold (~1e-7)
    print("Running detection on Hard X-rays (HEL1OS proxy)...")
    hard_events = detect_flares(df['xrsa'], threshold=1e-7, prominence=5e-8, distance=60, channel_name='Hard (HEL1OS proxy)')
    
    # Merge into a single master catalogue
    master = pd.concat([soft_events, hard_events], ignore_index=True)
    if not master.empty:
        master = master.sort_values(by='peak_time').reset_index(drop=True)
    
    out_dir = os.path.dirname(__file__)
    master_path = os.path.join(out_dir, "master_catalogue.csv")
    master.to_csv(master_path, index=False)
    
    print(f"\nGenerated Master Catalogue with {len(master)} total events.")
    print(f"  Soft X-ray (SoLEXS proxy) events: {len(soft_events)}")
    print(f"  Hard X-ray (HEL1OS proxy) events: {len(hard_events)}")
    print(f"Master catalogue saved to {master_path}")

if __name__ == "__main__":
    main()

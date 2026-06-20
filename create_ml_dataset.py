import pandas as pd
import numpy as np
import os
from sunpy.net import Fido
from sunpy.net import attrs as a
import sunpy.timeseries as ts

def load_data(tstart, tend):
    print(f"Loading cached GOES data for {tstart} to {tend}...")
    data_dir = os.path.expanduser("~/sunpy/data")
    
    # We specifically want GOES-15 1-minute average data to avoid mixing satellites
    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) 
             if f.endswith('.nc') and 'avg1m_g15' in f]
    
    if not files:
        raise ValueError(f"No GOES-15 files found in {data_dir}. Fido download required.")
        
    print(f"Found {len(files)} GOES-15 files. Loading...")
    goes_ts = ts.TimeSeries(files, concatenate=True)
    df = goes_ts.to_dataframe()
    return df

def main():
    # 1. Download a broader dataset (Sept 2017 active region)
    tstart = "2017-09-04"
    tend = "2017-09-10"
    df = load_data(tstart, tend)
    
    # We will use 'xrsb' (1-8 A) as SoLEXS Proxy (Soft X-rays)
    # We will use 'xrsa' (0.5-4 A) as HEL1OS Proxy (Hard X-rays)
    if 'xrsa' not in df.columns or 'xrsb' not in df.columns:
        print("Warning: Expected 'xrsa' and 'xrsb' in dataframe columns. Found:", df.columns)
        cols = df.columns[:2]
        df = df[cols].copy()
        df.columns = ['xrsa', 'xrsb']
    else:
        df = df[['xrsa', 'xrsb']].copy()
    
    # 2. Clean & Resample
    print("Resampling and removing missing values...")
    df = df.resample('1min').mean()
    df['xrsa'] = df['xrsa'].interpolate(method='linear')
    df['xrsb'] = df['xrsb'].interpolate(method='linear')
    df = df.dropna()
    
    print(f"Data shape after cleaning: {df.shape}")
    
    # Save the cleaned raw dataframe for the Nowcasting module to use
    out_dir = os.path.dirname(__file__)
    raw_csv_path = os.path.join(out_dir, "raw_flux_data.csv")
    df.to_csv(raw_csv_path)
    print(f"Saved raw flux data to {raw_csv_path}")
    
    # Feature engineering for BOTH channels
    print("Creating dual-channel features...")
    feature_cols = []
    for col in ['xrsa', 'xrsb']:
        df[f'{col}_ma_5'] = df[col].rolling(window=5, min_periods=1).mean()
        df[f'{col}_ma_15'] = df[col].rolling(window=15, min_periods=1).mean()
        df[f'{col}_diff'] = df[col].diff().fillna(0)
        df[f'{col}_std'] = df[col].rolling(window=15, min_periods=1).std().fillna(0)
        df[f'{col}_max'] = df[col].rolling(window=15, min_periods=1).max()
        
        feature_cols.extend([col, f'{col}_ma_5', f'{col}_ma_15', f'{col}_diff', f'{col}_std', f'{col}_max'])
    
    # 3. Normalize data
    print("Normalizing data...")
    epsilon = 1e-10
    df['raw_xrsb'] = df['xrsb'] # Keep for target classification
    
    pos_features = []
    for col in ['xrsa', 'xrsb']:
        pos_features.extend([col, f'{col}_ma_5', f'{col}_ma_15', f'{col}_max', f'{col}_std'])
        
    for col in pos_features:
        df[col] = np.log10(df[col] + epsilon)
        
    normalized_cols = []
    for col in feature_cols:
        norm_col = f'{col}_normalized'
        mean_val = df[col].mean()
        std_val = df[col].std()
        df[norm_col] = (df[col] - mean_val) / (std_val if std_val > 0 else 1.0)
        normalized_cols.append(norm_col)
    
    # 4. Create sliding windows
    window_size = 360   # 6 hours input
    forecast_horizon = 60  # predict peak in next 1 hour
    
    X = []
    y_class = []
    y_impact = []
    
    raw_flux_values = df['raw_xrsb'].values
    np.random.seed(42)
    
    print("Creating sliding windows for multiclass dataset...")
    for i in range(window_size, len(df) - forecast_horizon):
        window_x = df[normalized_cols].iloc[i-window_size : i].values.flatten()
        
        window_y_raw = raw_flux_values[i : i+forecast_horizon]
        max_flux = np.max(window_y_raw)
        
        if max_flux >= 1e-4:
            flare_class = 3 # X-class
        elif max_flux >= 1e-5:
            flare_class = 2 # M-class
        elif max_flux >= 1e-6:
            flare_class = 1 # C-class
        else:
            flare_class = 0 # B-class (or lower)
            
        # Synthesize spatial & CME features
        # Higher flare class -> more extreme values
        if flare_class == 3:
            cme_width = np.random.uniform(200, 360)
            cme_speed = np.random.uniform(1000, 3000)
            mag_class = np.random.choice([2, 3], p=[0.2, 0.8])
        elif flare_class == 2:
            cme_width = np.random.uniform(100, 250)
            cme_speed = np.random.uniform(600, 1500)
            mag_class = np.random.choice([1, 2], p=[0.5, 0.5])
        elif flare_class == 1:
            cme_width = np.random.uniform(30, 120)
            cme_speed = np.random.uniform(300, 800)
            mag_class = np.random.choice([0, 1], p=[0.7, 0.3])
        else:
            cme_width = np.random.uniform(0, 50)
            cme_speed = np.random.uniform(200, 400)
            mag_class = 0
            
        ar_lat = np.random.uniform(-40, 40)
        ar_lon = np.random.uniform(-90, 90)
        
        # Calculate Target: Earth Impact Probability
        # Higher probability if width is large (Halo) and longitude is near center (facing Earth)
        lon_factor = max(0, 1.0 - abs(ar_lon)/90.0) # 1.0 at center, 0.0 at edge
        width_factor = min(1.0, cme_width / 360.0)
        
        # Base probability depends heavily on CME actually being directed at us (width and lon)
        impact_prob = lon_factor * width_factor
        # Add some noise
        impact_prob = np.clip(impact_prob + np.random.normal(0, 0.05), 0.0, 1.0)
        if flare_class == 0:
            impact_prob = 0.0 # B-class or lower doesn't cause significant Earth impact
            
        # Combine flattened timeseries with scalars
        spatial_features = np.array([ar_lat, ar_lon, mag_class, cme_width, cme_speed])
        full_x = np.concatenate([window_x, spatial_features])
        
        X.append(full_x)
        y_class.append(flare_class)
        y_impact.append(impact_prob)
        
    X = np.array(X)
    y_class = np.array(y_class)
    y_impact = np.array(y_impact)
    
    print(f"Generated {len(X)} samples. Feature shape: {X.shape}")
    print(f"Class distribution: B(0):{np.sum(y_class == 0)}, C(1):{np.sum(y_class == 1)}, M(2):{np.sum(y_class == 2)}, X(3):{np.sum(y_class == 3)}")
    
    np.save(os.path.join(out_dir, "X_data.npy"), X)
    np.save(os.path.join(out_dir, "y_labels.npy"), y_class)
    np.save(os.path.join(out_dir, "y_impact.npy"), y_impact)
    print(f"Dataset saved to X_data.npy, y_labels.npy, y_impact.npy")

if __name__ == "__main__":
    main()

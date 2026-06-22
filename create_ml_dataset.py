import pandas as pd
import numpy as np
import os

def load_data():
    base_dir = os.path.dirname(__file__)
    solexs_path = os.path.join(base_dir, "raw_flux_data.csv")
    hel1os_path = os.path.join(base_dir, "raw_hel1os_data.csv")
    
    print(f"Loading SoLEXS data from {solexs_path}...")
    df_solexs = pd.read_csv(solexs_path, parse_dates=['time'], index_col='time')
    
    print(f"Loading HEL1OS data from {hel1os_path}...")
    df_hel1os = pd.read_csv(hel1os_path, parse_dates=['time'], index_col='time')
    
    # Scale SoLEXS counts to approximate GOES W/m^2 flux levels
    df_solexs['xrsa'] = df_solexs['xrsa'] * 2.7e-7
    df_solexs['xrsb'] = df_solexs['xrsb'] * 2.7e-7
    df_solexs = df_solexs[['xrsa', 'xrsb']].copy()
    
    # Extract CZT and CdTe channels from HEL1OS
    df_hel1os = df_hel1os[['hel1os_czt', 'hel1os_cdte']].copy()
    
    # Merge SoLEXS and HEL1OS data on time
    print("Merging SoLEXS and HEL1OS datasets...")
    df = pd.merge(df_solexs, df_hel1os, left_index=True, right_index=True, how='inner')
    return df

def main():
    # 1. Load combined dataset
    df = load_data()
    
    # 2. Clean & Resample within contiguous blocks
    print("Identifying contiguous observation blocks...")
    df = df.sort_index()
    diffs = df.index.to_series().diff() / pd.Timedelta(minutes=1)
    new_blocks = diffs > 15
    df['block_id'] = new_blocks.cumsum()
    
    core_cols = ['xrsa', 'xrsb', 'hel1os_czt', 'hel1os_cdte']
    cleaned_blocks = []
    
    for block_id, block in df.groupby('block_id'):
        if len(block) < 420: # 360 (window) + 60 (forecast)
            continue
        # Resample this block to continuous 1-minute cadence
        block_clean = block.drop(columns=['block_id']).resample('1min').mean()
        for col in core_cols:
            block_clean[col] = block_clean[col].interpolate(method='linear')
        block_clean = block_clean.dropna()
        
        if len(block_clean) >= 420:
            # Feature engineering for all channels LOCALLY within the block
            feature_cols = []
            for col in core_cols:
                block_clean[f'{col}_ma_5'] = block_clean[col].rolling(window=5, min_periods=1).mean()
                block_clean[f'{col}_ma_15'] = block_clean[col].rolling(window=15, min_periods=1).mean()
                block_clean[f'{col}_diff'] = block_clean[col].diff().fillna(0)
                block_clean[f'{col}_std'] = block_clean[col].rolling(window=15, min_periods=1).std().fillna(0)
                block_clean[f'{col}_max'] = block_clean[col].rolling(window=15, min_periods=1).max()
                
                feature_cols.extend([col, f'{col}_ma_5', f'{col}_ma_15', f'{col}_diff', f'{col}_std', f'{col}_max'])
                
            block_clean['block_id'] = block_id
            cleaned_blocks.append(block_clean)
            
    if not cleaned_blocks:
        print("ERROR: No valid contiguous blocks found!")
        return
        
    # Concatenate all blocks to normalize features globally
    df_all = pd.concat(cleaned_blocks)
    df_all['raw_xrsb'] = df_all['xrsb'] # Keep for target classification
    
    # 3. Normalize data
    print("Normalizing data globally...")
    epsilon = 1e-10
    pos_features = []
    for col in core_cols:
        pos_features.extend([col, f'{col}_ma_5', f'{col}_ma_15', f'{col}_max', f'{col}_std'])
        
    for col in pos_features:
        df_all[col] = np.log10(df_all[col] + epsilon)
        
    normalized_cols = []
    for col in feature_cols:
        mean_val = df_all[col].mean()
        std_val = df_all[col].std()
        
        norm_col = f'{col}_normalized'
        df_all[norm_col] = (df_all[col] - mean_val) / (std_val if std_val > 0 else 1.0)
        normalized_cols.append(norm_col)
        
    # 4. Create sliding windows block by block
    window_size = 360
    forecast_horizon = 60
    step_size = 15
    
    X = []
    y_class = []
    y_impact = []
    
    np.random.seed(42)
    
    print("Creating sliding windows from blocks...")
    for block_id, block in df_all.groupby('block_id'):
        raw_flux_values = block['raw_xrsb'].values
        # Sliding window within this block
        for i in range(window_size, len(block) - forecast_horizon, step_size):
            window_x = block[normalized_cols].iloc[i-window_size : i].values.flatten()
            
            window_y_raw = raw_flux_values[i : i+forecast_horizon]
            max_flux = np.max(window_y_raw)
            
            if max_flux >= 1e-4:
                flare_class = 3 # X-class
            elif max_flux >= 1e-5:
                flare_class = 2 # M-class
            elif max_flux >= 1e-6:
                flare_class = 1 # C-class
            else:
                flare_class = 0 # B-class
                
            # Synthesize spatial & CME features
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
            
            # Target Earth Impact Probability
            lon_factor = max(0, 1.0 - abs(ar_lon)/90.0)
            width_factor = min(1.0, cme_width / 360.0)
            impact_prob = lon_factor * width_factor
            impact_prob = np.clip(impact_prob + np.random.normal(0, 0.05), 0.0, 1.0)
            if flare_class == 0:
                impact_prob = 0.0
                
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
    
    out_dir = os.path.dirname(__file__)
    np.save(os.path.join(out_dir, "X_data.npy"), X)
    np.save(os.path.join(out_dir, "y_labels.npy"), y_class)
    np.save(os.path.join(out_dir, "y_impact.npy"), y_impact)
    print(f"Dataset saved to X_data.npy, y_labels.npy, y_impact.npy")

if __name__ == "__main__":
    main()

import os
import glob
import zipfile
import pandas as pd
import numpy as np
from astropy.io import fits
import warnings

# Suppress FITS warnings for slightly non-compliant FITS standard headers
warnings.filterwarnings('ignore', category=UserWarning, append=True)

import tempfile

def extract_lc(zfile, fname):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".gz") as tmp:
        tmp.write(zfile.read(fname))
        tmp_path = tmp.name
        
    try:
        with fits.open(tmp_path) as hdul:
            if 'RATE' in hdul:
                data = hdul['RATE'].data
                # TIME is UNIX timestamp in seconds
                times = data['TIME']
                counts = data['COUNTS']
                return pd.DataFrame({'time': pd.to_datetime(times, unit='s'), 'counts': counts})
    finally:
        os.remove(tmp_path)
    return None

def process_solexs_data(data_dir):
    zip_files = glob.glob(os.path.join(data_dir, "*.zip"))
    print(f"Found {len(zip_files)} zip files.")
    
    all_data = []
    
    for i, zip_path in enumerate(sorted(zip_files)):
        print(f"[{i+1}/{len(zip_files)}] Processing {os.path.basename(zip_path)}...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as z:
                # Find the lc.gz files
                lc_files = [f for f in z.namelist() if f.endswith('.lc.gz')]
                sdd1_file = next((f for f in lc_files if 'SDD1' in f), None)
                sdd2_file = next((f for f in lc_files if 'SDD2' in f), None)
                
                df_sdd1 = None
                df_sdd2 = None
                
                if sdd1_file:
                    try:
                        df_sdd1 = extract_lc(z, sdd1_file)
                    except Exception as e:
                        pass
                        
                if sdd2_file:
                    try:
                        df_sdd2 = extract_lc(z, sdd2_file)
                    except Exception as e:
                        pass

                # Merge SDD1 and SDD2 for the day
                if df_sdd1 is not None and df_sdd2 is not None:
                    # Time indices usually match exactly, but let's outer join
                    df_day = pd.merge(df_sdd1, df_sdd2, on='time', how='outer', suffixes=('_sdd1', '_sdd2'))
                elif df_sdd1 is not None:
                    df_day = df_sdd1.rename(columns={'counts': 'counts_sdd1'})
                    df_day['counts_sdd2'] = df_day['counts_sdd1'] # duplicate as fallback
                elif df_sdd2 is not None:
                    df_day = df_sdd2.rename(columns={'counts': 'counts_sdd2'})
                    df_day['counts_sdd1'] = df_day['counts_sdd2'] # duplicate as fallback
                else:
                    continue
                    
                all_data.append(df_day)
        except Exception as e:
            print(f"Error processing {zip_path}: {e}")

    if not all_data:
        print("No valid data found.")
        return
        
    print("Concatenating all days (this may take a moment)...")
    full_df = pd.concat(all_data, ignore_index=True)
    
    # Sort and set index
    full_df = full_df.sort_values('time').set_index('time')
    
    # Resample to 1 minute
    print("Resampling to 1-minute cadence...")
    df_1m = full_df.resample('1min').mean()
    
    # Fill missing with interpolation
    df_1m = df_1m.interpolate(method='linear').fillna(0)
    
    # Rename columns to match existing pipeline
    df_1m = df_1m.rename(columns={'counts_sdd1': 'xrsa', 'counts_sdd2': 'xrsb'})
    
    out_path = os.path.join(os.path.dirname(__file__), "raw_flux_data.csv")
    df_1m.to_csv(out_path)
    print(f"Saved {len(df_1m)} 1-minute rows to {out_path}")

if __name__ == "__main__":
    process_solexs_data(r"C:\Users\harsh\Downloads\solexs_2026Jun21T182607399")

"""
Extract HEL1OS (High Energy L1 Orbiting X-ray Spectrometer) data from nested ZIP archives.

HEL1OS has two detector families:
  - CZT (Cadmium Zinc Telluride): Hard X-ray, energy bands 20-40, 40-60, 60-80, 80-150, 18-160 keV
  - CdTe (Cadmium Telluride): Softer X-ray, energy bands 5-20, 20-30, 30-40, 40-60, 1.8-90 keV

Each observation has light curves with columns: MJD, ISOT, CTR (count rate), STAT_ERR

This script extracts the broadband light curves from both detector families,
resamples to 1-minute cadence, and saves as raw_hel1os_data.csv.
"""

import os
import io
import sys
import zipfile
import numpy as np
import pandas as pd
from astropy.io import fits

# ── Configuration ──────────────────────────────────────────────
HEL1OS_DIR = r"C:\Users\harsh\Downloads\Hel10s"
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "raw_hel1os_data.csv")

# We extract these broadband light curves (the widest energy range per detector)
TARGET_BANDS = {
    "czt": {
        "file_pattern": "lightcurve_czt1.fits",
        "hdu_pattern": "18.00KEV_TO_160.00KEV",  # broadband CZT
        "column": "hel1os_czt",
    },
    "cdte": {
        "file_pattern": "lightcurve_cdte1.fits",
        "hdu_pattern": "1.80KEV_TO_90.00KEV",    # broadband CdTe
        "column": "hel1os_cdte",
    },
}

# Also extract sub-bands for richer feature set
SUB_BANDS = {
    "czt_20_40":  ("lightcurve_czt1.fits",  "20.00KEV_TO_40.00KEV",  "czt_20_40"),
    "czt_40_60":  ("lightcurve_czt1.fits",  "40.00KEV_TO_60.00KEV",  "czt_40_60"),
    "czt_60_80":  ("lightcurve_czt1.fits",  "60.00KEV_TO_80.00KEV",  "czt_60_80"),
    "czt_80_150": ("lightcurve_czt1.fits",  "80.00KEV_TO_150.00KEV", "czt_80_150"),
    "cdte_5_20":  ("lightcurve_cdte1.fits", "5.00KEV_TO_20.00KEV",   "cdte_5_20"),
    "cdte_20_30": ("lightcurve_cdte1.fits", "20.00KEV_TO_30.00KEV",  "cdte_20_30"),
}


def extract_lightcurve_from_fits(fits_bytes, hdu_pattern):
    """Extract a light curve from a FITS binary table matching the HDU pattern."""
    try:
        hdul = fits.open(io.BytesIO(fits_bytes), memmap=False)
        for hdu in hdul[1:]:
            if hdu_pattern in hdu.name:
                data = hdu.data
                times = pd.to_datetime([t.decode() if isinstance(t, bytes) else t for t in data["ISOT"]])
                counts = data["CTR"].astype(np.float64)
                errors = data["STAT_ERR"].astype(np.float64)
                hdul.close()
                return times, counts, errors
        hdul.close()
    except Exception as e:
        pass
    return None, None, None


def process_inner_zip(inner_zip_bytes):
    """Process one observation session (inner ZIP) and return DataFrames."""
    try:
        inner = zipfile.ZipFile(io.BytesIO(inner_zip_bytes), "r")
    except Exception:
        return None

    results = {}

    for name in inner.namelist():
        # Check broadband targets
        for key, cfg in TARGET_BANDS.items():
            if cfg["file_pattern"] in name:
                fits_bytes = inner.read(name)
                times, counts, errors = extract_lightcurve_from_fits(fits_bytes, cfg["hdu_pattern"])
                if times is not None:
                    results[cfg["column"]] = pd.Series(counts, index=times)
                    results[cfg["column"] + "_err"] = pd.Series(errors, index=times)

        # Check sub-bands
        for key, (file_pat, hdu_pat, col_name) in SUB_BANDS.items():
            if file_pat in name:
                fits_bytes = inner.read(name)
                times, counts, _ = extract_lightcurve_from_fits(fits_bytes, hdu_pat)
                if times is not None:
                    results[col_name] = pd.Series(counts, index=times)

    inner.close()

    if not results:
        return None

    df = pd.DataFrame(results)
    df.index.name = "time"
    return df


def main():
    all_frames = []
    outer_files = sorted([f for f in os.listdir(HEL1OS_DIR) if f.endswith(".zip")])
    
    print(f"Found {len(outer_files)} HEL1OS archive files.")
    
    session_count = 0
    for oi, outer_name in enumerate(outer_files, 1):
        outer_path = os.path.join(HEL1OS_DIR, outer_name)
        print(f"\n[{oi}/{len(outer_files)}] {outer_name}")

        try:
            outer = zipfile.ZipFile(outer_path, "r")
        except Exception as e:
            print(f"  Skipping (not a valid ZIP): {e}")
            continue

        inner_names = [n for n in outer.namelist() if n.endswith(".zip")]
        print(f"  Contains {len(inner_names)} observation sessions.")

        for ii, inner_name in enumerate(inner_names, 1):
            session_count += 1
            sys.stdout.write(f"\r  Processing session {ii}/{len(inner_names)}: {inner_name[:60]}...")
            sys.stdout.flush()

            inner_bytes = outer.read(inner_name)
            df = process_inner_zip(inner_bytes)
            if df is not None and len(df) > 0:
                all_frames.append(df)

        outer.close()
        print(f"\n  Done. Total sessions so far: {session_count}")

    if not all_frames:
        print("ERROR: No data extracted!")
        return

    print(f"\nConcatenating {len(all_frames)} sessions...")
    combined = pd.concat(all_frames)
    combined.sort_index(inplace=True)
    combined = combined[~combined.index.duplicated(keep="first")]

    print(f"Total raw rows: {len(combined)}")
    print(f"Time range: {combined.index.min()} to {combined.index.max()}")
    print(f"Columns: {list(combined.columns)}")

    # Resample to 1-minute cadence
    print("Resampling to 1-minute cadence...")
    resampled = combined.resample("1min").mean()
    resampled.dropna(how="all", inplace=True)

    print(f"Resampled rows: {len(resampled)}")

    resampled.index.name = "time"
    resampled.to_csv(OUTPUT_FILE)
    print(f"\nSaved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

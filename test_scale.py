import pandas as pd
import numpy as np

def main():
    s = pd.read_csv('raw_flux_data.csv', parse_dates=['time'], index_col='time')
    h = pd.read_csv('raw_hel1os_data.csv', parse_dates=['time'], index_col='time')
    m = pd.merge(s, h, left_index=True, right_index=True, how='inner')
    
    counts = m['xrsb'].values
    print("Max raw xrsb counts in overlap:", counts.max())
    
    # We look at the rolling max in next 60 minutes, which is what the labels are built on
    m_resampled = m.resample('1min').mean()
    m_resampled['xrsb'] = m_resampled['xrsb'].interpolate(method='linear')
    m_resampled = m_resampled.dropna()
    
    raw_flux_values = m_resampled['xrsb'].values
    forecast_horizon = 60
    step_size = 15
    
    max_fluxes = []
    for i in range(360, len(m_resampled) - forecast_horizon, step_size):
        window_y_raw = raw_flux_values[i : i+forecast_horizon]
        max_fluxes.append(np.max(window_y_raw))
        
    max_fluxes = np.array(max_fluxes)
    
    for scale in [1e-10, 1e-9, 1e-8, 1e-7, 2.7e-7]:
        scaled = max_fluxes * scale
        c3 = np.sum(scaled >= 1e-4)
        c2 = np.sum((scaled >= 1e-5) & (scaled < 1e-4))
        c1 = np.sum((scaled >= 1e-6) & (scaled < 1e-5))
        c0 = np.sum(scaled < 1e-6)
        print(f"Scale {scale:.1e} -> X: {c3}, M: {c2}, C: {c1}, B: {c0}")

if __name__ == '__main__':
    main()

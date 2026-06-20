import matplotlib.pyplot as plt
from sunpy.net import Fido
from sunpy.net import attrs as a
import sunpy.timeseries as ts
import os

def main():
    print("Searching for GOES XRS data for the X9.3 flare on Sept 6, 2017...")
    # Time range around the X9.3 flare
    tstart = "2017-09-06 11:30"
    tend = "2017-09-06 12:30"
    
    # Query GOES XRS data
    result = Fido.search(a.Time(tstart, tend), a.Instrument("XRS"))
    print(f"Search results:\n{result}")
    
    # Download the data
    print("Downloading data...")
    downloaded_files = Fido.fetch(result)
    
    if not downloaded_files:
        print("Failed to download GOES data.")
        return
        
    print(f"Downloaded files: {downloaded_files}")
    
    # Load into a TimeSeries
    print("Loading data into SunPy TimeSeries...")
    goes_ts = ts.TimeSeries(downloaded_files)
    
    # SunPy TimeSeries can sometimes be a list if multiple files are loaded
    if isinstance(goes_ts, list):
        goes_ts = ts.TimeSeries(goes_ts, concatenate=True)
    
    # Plot the light curve
    print("Generating plot...")
    fig, ax = plt.subplots(figsize=(10, 5))
    goes_ts.plot(axes=ax)
    
    plt.title("GOES X-Ray Flux - Sept 6, 2017 (X9.3 Flare)")
    plt.ylabel("Watts m$^{-2}$")
    plt.xlabel("Time (UTC)")
    plt.grid(True)
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(os.path.dirname(__file__), "flare_plot.png")
    plt.savefig(plot_path)
    print(f"Plot saved successfully to: {plot_path}")

if __name__ == "__main__":
    main()

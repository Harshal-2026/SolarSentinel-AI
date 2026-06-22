import sys
from astropy.io import fits

filepath = sys.argv[1]
with fits.open(filepath) as hdul:
    hdul.info()
    
    if len(hdul) > 1:
        print("\n--- Extension 1 Header ---")
        print(repr(hdul[1].header))
        print("\n--- Extension 1 Columns ---")
        print(hdul[1].columns)

# Spectrum_Plot
This python script (line_id.py)

- computes an average spectrum from a datacube
- determines the noise in the average spectrum in a line-free range
- creates a figure of the average spectrum
- lines can be annotated with corresponding molecule

This script can be used to iteratively identify emission/absorption lines above a defined threshold (e.g. a signal-to-noise ratio > 5). 

Required Input Files:
- .fits file of the datacube
- TransitionProperties.dat

    first column: rest-frequency (GHz)
    
    second column: molecule label


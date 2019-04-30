# Line_ID_Plot
Required Input Files:
- .fits file of the datacube
- TransitionProperties.dat
    first column: rest-frequency (GHz)
    second column: molecule label
    
    other columns can be ignored

line_id.py:
- extract spectrum (spatially averaged) from a datacube
- determine noise in spectrum
- plot spectrum with labelled lines

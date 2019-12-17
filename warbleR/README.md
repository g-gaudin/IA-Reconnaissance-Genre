# WarbleR Python Script #

### Requirements: ###
- Python 3.7.5
- R 3.6.2

 **rpy2 is not required as Python calls Rscript subprocess directly to run R scripts**

### R scripts: ###
- packageInstalleR.R - Install all necessary R packages (warbleR, Rraven and devtools)
- packageCheckeR.R - Check required packages are installed and list all available warbleR functions
- packageRemoveR.R - Uninstall all previously installed packages
- voiceAnalyzeR.R - Analyze all voice samples in wav directory and save all required voice parameters in *results.csv* file

### Python scripts: ###
- rPackages.py - Installs and then checks all packages required to run WarbleR
- warbleR.py - main script that runs voice analyzer R script and stores all data in variable *table*

### Results: ###
- *-p1.tiff - a long spectrogram of each voice sample
- *-autodetec.ls-th15-env.abs-bp0.08.0.3-smo600-midu0.1-mxdu-pw1-p1.jpeg - a spectogram of each voice sample with marked automatically detected gaps based on amplitude, duration, and frequency range attributes.
- result.csv - a table with all extracted voice sample parameters

### IMPORTANT: In rPackages.py and warbleR.py change *path* variable so it should correspond to /your/path/to/WarbleR directory ###
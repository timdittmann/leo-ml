# gnssroML
This repository was used to develop a machine-learning based multiclass classifier of GNSS Radio Occultation (GNSS-RO) measurements from the Spire GNSS-RO constellation made available through the NOAA Commercial Data Program and UCAR/NOAA Science Collaboration Program.  Primary classes of interest include ionospheric scintillation and radio frequency interference.

For further scientific context, this repository is published in conjunction with the following manuscript:
>Dittmann, Chang, and Morton (202?) **Machine Learning Classification of Ionosphere and RFI Disturbances in Spaceborne GNSS Radio Occultation Measurements.**


**Open Software:**
This manuscript's analysis is covered in `notebooks/navigation_2024_manuscript.ipynb`.  The authors also wish to acknowledge the openly-developed software used in this analysis: Numpy, Pandas, Xarray, Scikit-learn, XGBoost, and matplotlib.

**Open Data**
This manuscript's feature sets are hosted at [Zenodo](TODO: Zenodo link).

**Acknowledgements**
The data used in this analysis was made possible by the NOAA Commercial Data Program and UCAR/NOAA Science Collaboration Program.   

**Environment**
To reproduce this analysis, we suggest creating a new python virtual environment and building the environment from the `src/pyproject.toml` file, something like this:  (after cloning in the repo and changing into the repo root directory)

```
pip install virtualenv
virtualenv -p python3 .venv
source .venv/bin/activate
pip install -e src/
```
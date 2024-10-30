This repository is published in conjunction with 

Dittmann, Chang, and Morton.  

**Open Software:**
This manuscript's analysis is covered in `notebooks/navigation_2024_manuscript.ipynb`.  

**Open Data**
This manuscript's feature sets are hosted at Zenodo.
DOI

**Acknowledgements**
The data used in this analysis was made possible by the NOAA Commercial Data Program and UCAR/NOAA Science Collaboration Program.  The authors also wish to acknowledge the openly-developed software used in this analysis: Numpy, Pandas, Xarray, Scikit-learn, XGBoost, and matplotlib. 

**Environment**
To reproduce this analysis, we suggest creating a new python virtual environment and building the environment from the `src/pyproject.toml` file, something like this:  (after cloning in the repo and changing into the repo root directory)

```
pip install virtualenv
virtualenv -p python3 .venv
source .venv/bin/activate
pip install -e src/
```
# GNSS-LEO Notebook Summary
Effectively chronologic summary, if it helps with context (or at least for me to remember :smile:)
1. initial experiment:

* label_ro_nb.ipynb
    * convert `data/chang_labels.txt` to `../data/converted_labels.pkl`
    * `make_label_plots.py` used to generate feature sets and plot for 2023.144
    * Used to read in file and where user assigns labels
* view_ro_for_labels
    * used to rotate through file and provide visual for analyst to label in `label_ro_nb`
* `ml-pipeline.ipynb`
    * train/validate/test model

2. identifying comms overlaps
* ro-comms-overlap.ipynb

3. labeling data from comms overlap
* comms_rfi.ipynb
* label_ro_nb.ipynb
    * Used to read in file and where user assigns labels, added in comms file (`../data/converted_labels_comms.pkl`)
* view_ro_for_labels
    * used to rotate through file and provide visual for analyst to label in `label_ro_nb`
* `ml-pipeline-comms.ipynb`
    * train/validate/test model with new comms labels

4. finding more scint. samples for training/testing
* more_scint_samples.ipynb
    * generates `../data/converted_labels_scint_v2.pkl`
* `ml-pipeline-scint.ipynb`
    * train/validate/test model with more scintillation

5. finding more rfi samples for training/testing
* `more_rfi_samples.ipynb`
    * generates `../data/converted_labels_rfi2_v2.pkl`
* `ml-pipeline-multi.ipynb`
    * train/validate/test MULTI CLASS model with more rfi samples



* plot_prof_dist.ipynb




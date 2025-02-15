#!/bin/bash
# G. Hayes 2024
# This script is used to batch preprocess the tcd data by looping through parameters for each subject using "1_tcd_ramp_preprocessing.ipynb" for the analysis presented in:
# G. Hayes, S. Sparks, J. Pinto, and D. P. Bulte, “Ramp protocol for non-linear cerebrovascular reactivity with transcranial doppler ultrasound,” Journal of Neuroscience Methods, vol. 416, p. 110381, Apr. 2025, doi: 10.1016/j.jneumeth.2025.110381.
# To run the fitting of the data after ther preprocessing, use the "2_tcd_ramp_fitting.ipynb" script.
# Updated this script for your purposes, notably:
# - the data file name
# - the output file name
# - alter parameters that may differ for your data
papermill --version

# Example subject 1
filename='sub-001_ses-01_dat-YYYYMMDD_task-ramp_pwl.txt'
O2prominence='0.8'
comment_start='0'
comment_end='9'
P_oxford='1017.7'
man_shift='0'
peak_dif_thresh='0.4'

papermill tcd_ramp_preprocessing.ipynb tcd_ramp_preprocessing_002.ipynb -p filename $filename -p O2prominence $O2prominence -p comment_start $comment_start -p comment_end $comment_end -p P_oxford $P_oxford -p man_shift $man_shift -p peak_dif_thresh $peak_dif_thresh

# Example subject 2
filename='sub-002_ses-01_dat-YYYYMMDD_task-ramp_pwl.txt'
O2prominence='1'
comment_start='0'
comment_end='7'
P_oxford='1009.8'
man_shift='0'
peak_dif_thresh='0.4'

papermill tcd_ramp_preprocessing.ipynb tcd_ramp_preprocessing_015.ipynb -p filename $filename -p O2prominence $O2prominence -p comment_start $comment_start -p comment_end $comment_end -p P_oxford $P_oxford -p man_shift $man_shift -p peak_dif_thresh $peak_dif_thresh
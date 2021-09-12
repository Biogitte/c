#!/bin/bash

# Working directory - remember to specify accordingly.
export WORK_DIR=~/Desktop/code/data_science

# Data directories
export DATA=$WORK_DIR/data
export DATA_RAW=$DATA/raw
export DATA_PROC=$DATA/processed

# Source code directory
export SRC=$WORK_DIR/src

# Executables directory
export EXECUTE=$WORK_DIR/executables

# Jupyter Notebooks directory
export NOTEBOOKS=$WORK_DIR/notebooks

# Variables related to fetching Kaggle data
export FETCH_KAGGLE_DATA=$SRC/fetch_kaggle_data.py # the python script
export KAGGLE_DATASET='fedesoriano/stroke-prediction-dataset' # the Kaggle dataset to download
export KAGGLE_NEW_NAME='stroke_raw.csv' # the new name of the Kaggle data set

# Variables related to data cleaning and preparation
export CLEAN_DATA_SCRIPT=$SRC/clean_data.py # the python script
export INPUT_FILE_PATTERN=$DATA_RAW'/*_'$KAGGLE_NEW_NAME # filename pattern with date as wild-card
export UNWANTED_COLS='id,age,bmi' # columns to remove (like this: 'id,age,bmi')
export LABEL_COL='stroke' # the column with classification labels
export CLEAN_THE_DATA=$EXECUTE'/data_prep.sh' # the bash script to clean and prepare the data




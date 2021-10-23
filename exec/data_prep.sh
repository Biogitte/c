#!/bin/bash

echo 'Downloading Kaggle dataset:' "$KAGGLE_DATASET"

python3 "$FETCH_KAGGLE_DATA" "$KAGGLE_DATASET" "$DATA_RAW" "$KAGGLE_NEW_NAME"

echo 'Download completed. The file can be found here:' "$DATA_RAW"'/'"$KAGGLE_NEW_NAME"

echo 'Initialising data cleaning and encoding...'

python3 "$CLEAN_DATA_SCRIPT" "$INPUT_FILE_PATTERN" "$DATA_PROC" "$LABEL_COL"

echo 'Data cleaning and encoding completed.'

echo 'Find QC HTML reports, cleaned and encoded datasets in this directory:' "$DATA_PROC"



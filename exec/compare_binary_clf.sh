#!/bin/bash

echo 'Comparing binary classifiers...'

python3 "$BINARY_CLFS_SCRIPT" '--df' "$DATA_PROC" '--label_col' "$LABEL_COL" '--out_dir' "$DATA_PROC"

echo 'Completed comparing binary classifiers.'
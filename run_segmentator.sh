#!/bin/bash

# Check if a directory argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <directory>"
    exit 1
fi

# Base directory containing the scene folders
BASE_DIR="$1"

# Iterate over each _vh_clean_2.ply file in the directory structure
find "$BASE_DIR" -type f -name "*_vh_clean_2.ply" | while read -r INPUT_FILE; do
    # Execute the segmentator command with the current input file
    echo "Processing file: $INPUT_FILE"
    ./ScanNet/Segmentator/segmentator "$INPUT_FILE" 0.01 20
done


##!/bin/bash
#
## Base directory containing the scene folders
#BASE_DIR="/home/zihui/HDD/ScanNetv2/scans_test"
#
## Iterate over each _vh_clean_2.ply file in the directory structure
#find "$BASE_DIR" -type f -name "*_vh_clean_2.ply" | while read -r INPUT_FILE; do
#    # Execute the segmentator command with the current input file
#    echo "Processing file: $INPUT_FILE"
#    ./ScanNet/Segmentator/segmentator "$INPUT_FILE" 0.01 20
#done
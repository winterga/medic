#!/bin/bash

# Check if the input folder is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <input_folder>"
    exit 1
fi

input_folder="$1"

# Ensure crop.py and resize_512.py exist
crop_script="crop.py"
resize_script="resize_512.py"

if [ ! -f "$crop_script" ]; then
    echo "Error: $crop_script not found!"
    exit 1
fi

if [ ! -f "$resize_script" ]; then
    echo "Error: $resize_script not found!"
    exit 1
fi

# Find all images and process them in parallel
find "$input_folder" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | \
xargs -P 16 -I {} bash -c '  
    echo "Processing {}"
    python3 crop.py "{}" && python3 resize_512.py "{}" && echo "Successfully processed {}" || echo "Error processing {}"
'

echo "All images processed."
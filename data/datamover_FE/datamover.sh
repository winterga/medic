#!/bin/bash

# Check if the input folder is passed as an argument
if [ -z "$1" ]; then
    echo "Usage: $0 <input_folder>"
    exit 1
fi

input_folder="$1"

# Ensure crop.py and resize_512.py exist in the current directory or provide the full path if necessary
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

# Find all image files recursively in the input folder (supports jpg, jpeg, png)
find "$input_folder" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | while read -r image; do
    # echo "Processing $image"

    # Call the crop.py script on the image
    python3 "$crop_script" "$image"
    
    if [ $? -ne 0 ]; then
        echo "Error occurred during cropping for $image"
        continue
    fi

    # Call the resize_512.py script on the cropped image
    python3 "$resize_script" "$image"

    if [ $? -ne 0 ]; then
        echo "Error occurred during resizing for $image"
        continue
    fi

    echo "Successfully processed $image"
done

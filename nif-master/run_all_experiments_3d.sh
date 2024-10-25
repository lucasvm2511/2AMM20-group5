#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <config_dir> <image_dir> <root_all>"
    exit 1
fi

# Input arguments
CONFIG_DIR="$1"
IMAGE_DIR="$2"
ROOT_ALL="$3"

# Create the root results directory if it doesn't exist
mkdir -p "$ROOT_ALL"

# Loop through all YAML configuration files
for config in "$CONFIG_DIR"/*.yaml; do
    # Extract the filename without the extension (for result root)
    config_name=$(basename "$config" .yaml)
    
    # Loop through all image files (assuming .npy files)
    for image in "$IMAGE_DIR"/*.npy; do
        # Extract the first 2 characters of the image filename
        image_prefix=$(basename "$image" .npy | cut -c 1-2)
        
        # Construct the result_root by combining config_name and image_prefix
        result_root="${config_name}_${image_prefix}"
        
        echo "Training model with config $config on image $image, storing results in $ROOT_ALL/$result_root"
        
        # Call your existing experiment script with the correct order of arguments: config, image, and result_root
        ./nif-master/experiment_3d.sh "$config" "$image" "$ROOT_ALL/$result_root"
        
        # Check if the training failed
        if [ $? -ne 0 ]; then
            echo "Training failed for config $config and image $image"
            exit 1
        fi
    done
done

echo "Training completed for all configurations and images."

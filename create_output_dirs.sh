#!/bin/bash

# Find all yaml files in configs/ and create corresponding output directories
for config in configs/*.yaml; do
    if [ -f "$config" ]; then
        # Extract config name without path and extension
        config_name=$(basename "$config" .yaml)
        # Create output directory if it doesn't exist
        mkdir -p "outputs/${config_name}"
    fi
done

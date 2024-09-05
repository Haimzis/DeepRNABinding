#!/bin/bash

# Define directories and base filenames
SEQUENCES_FILE="RNAcompete_sequences_rc.txt"
HTR_SELEX_DIR="htr-selex"
PYTHON_SCRIPT="main.py"
OUTPUT_DIR="output"

export PYTHONUNBUFFERED=1

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Function to process a single RBP
process_rbp() {
    local RBP_NUM=$1
    local HTR_SELEX_FILES=()
    
    # Collect relevant HTR-SELEX files
    for CYCLE in {1..4}; do
        FILE="${HTR_SELEX_DIR}/RBP${RBP_NUM}_${CYCLE}.txt"
        if [ -f "$FILE" ]; then
            FILENAME=$(basename "$FILE")
            HTR_SELEX_FILES+=("$FILENAME")
        fi
    done

    # Check if any HTR-SELEX files were found
    if [ ${#HTR_SELEX_FILES[@]} -gt 0 ]; then
        echo "Processing RBP${RBP_NUM} with files: ${HTR_SELEX_FILES[@]}"
        
        # Construct the command to run
        CMD="python -u $PYTHON_SCRIPT $SEQUENCES_FILE ${HTR_SELEX_FILES[@]}"
        
        # Print the command
        echo "Running command: $CMD"
        
        # Run the command and redirect output to a log file
        $CMD > "${OUTPUT_DIR}/RBP${RBP_NUM}.txt" 2>&1
        
        # Rename the binding intensities file after the run, if it exists
        if [ -f "bindings_intensities.txt" ]; then
            mv bindings_intensities.txt "${OUTPUT_DIR}/RBP${RBP_NUM}.txt"
        else
            echo "Warning: bindings_intensities.txt not found for RBP${RBP_NUM}."
        fi
    else
        echo "No files found for RBP${RBP_NUM}. Skipping..."
    fi
}

# Loop through all RBP numbers and run each sequentially
for RBP_NUM in {1..38}; do
    process_rbp $RBP_NUM
done

echo "All RBP processing tasks are complete."

echo "All test results have been saved in the output directory."

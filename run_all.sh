#!/bin/bash

# Define directories and base filenames
SEQUENCES_FILE="RNAcompete_sequences_rc.txt"
HTR_SELEX_DIR="data/htr-selex"
PYTHON_SCRIPT="main.py"

export PYTHONUNBUFFERED=1

# Function to process a single RBP
process_rbp() {
    local RBP_NUM=$1
    local HTR_SELEX_FILES=()
    
    # Collect relevant HTR-SELEX files
    for CYCLE in {1..4}; do
        FILE="${HTR_SELEX_DIR}/RBP${RBP_NUM}_${CYCLE}.txt"
        if [ -f "$FILE" ]; then
            # Use basename to extract just the filename
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
        $CMD > "output_RBP${RBP_NUM}.log" 2>&1 &
    else
        echo "No files found for RBP${RBP_NUM}. Skipping..."
    fi
}

# Loop through all RBP numbers and run each in parallel
for RBP_NUM in {1..38}; do
    process_rbp $RBP_NUM
done

# Wait for all background jobs to finish
wait

echo "All RBP processing tasks are complete."

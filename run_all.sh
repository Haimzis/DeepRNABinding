#!/bin/bash

# Define directories and base filenames
SEQUENCES_FILE="RNAcompete_sequences_rc.txt"
HTR_SELEX_DIR="data/htr-selex"
PYTHON_SCRIPT="main.py"
OUTPUT_DIR="output"
COMBINED_FILE="combined_correlations.txt"
MAX_PROCESSES=10

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
        $CMD > "${OUTPUT_DIR}/output_RBP${RBP_NUM}.log" 2>&1 &
    else
        echo "No files found for RBP${RBP_NUM}. Skipping..."
    fi
}

# Function to wait for jobs to finish if the maximum number of processes are running
wait_for_processes() {
    while [ $(jobs -r | wc -l) -ge $MAX_PROCESSES ]; do
        sleep 1
    done
}

# Loop through all RBP numbers and run each in parallel
for RBP_NUM in {1..38}; do
    wait_for_processes
    process_rbp $RBP_NUM
done

# Wait for all background jobs to finish
wait

echo "All RBP processing tasks are complete."

# Combine all correlation results into a single file
echo "INFO:root:Computed correlations:" > $COMBINED_FILE

for RBP_NUM in {1..38}; do
    LOG_FILE="${OUTPUT_DIR}/output_RBP${RBP_NUM}.log"
    if [ -f "$LOG_FILE" ]; then
        # Extract numeric correlation values and append to the combined file
        grep -Eo '[+-]?[0-9]+\.[0-9]+' "$LOG_FILE" >> $COMBINED_FILE
    fi
done

# Calculate the average correlation and append to the combined file
average=$(grep -Eo '[+-]?[0-9]+\.[0-9]+' $COMBINED_FILE | awk '{sum+=$1} END {print sum/NR}')
echo "INFO:root:avg correlation: $average" >> $COMBINED_FILE

# Remove individual log files
rm -f ${OUTPUT_DIR}/output_RBP*.log

echo "Combined results are saved in $COMBINED_FILE and individual log files have been removed."

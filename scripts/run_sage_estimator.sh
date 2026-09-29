#!/bin/bash

# Try to use conda if available
if command -v conda &> /dev/null; then
    # Source conda initialization
    CONDA_BASE=$(conda info --base 2>/dev/null)
    if [ -n "$CONDA_BASE" ]; then
        source "$CONDA_BASE/etc/profile.d/conda.sh" 2>/dev/null
        
        # Try to activate sage environment (don't fail if it doesn't exist)
        conda activate sage 2>/dev/null
    fi
fi

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Run the sage estimator script with arguments
python "$SCRIPT_DIR/sage_sis_estimator.py" "$@"

# Capture the exit code
EXIT_CODE=$?

# Deactivate conda environment if it was activated
if command -v conda &> /dev/null; then
    conda deactivate 2>/dev/null
fi

exit $EXIT_CODE

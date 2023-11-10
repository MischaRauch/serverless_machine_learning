#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Print commands and their arguments as they are executed.
set -x

# Define the base directory where the script is located.
BASEDIR=$(dirname "$0")

# Activate a virtual environment if needed
# source "$BASEDIR/venv/bin/activate"

# Or if your Python script reads the API key from an environment variable:
export HOPSWORKS_API_KEY=$API_KEY

# Execute a Python script, for instance, to train a model using the Iris dataset
python "$BASEDIR/train_iris_model.py"

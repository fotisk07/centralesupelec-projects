#!/bin/bash
# ---------------------------------------------------------
# Submit a file to the Kaggle "Beacon Biosignals Sleep Staging 2025" competition
# Usage: ./submit_kaggle.sh <submission_file> "<submission_message>"
# Example: ./submit_kaggle.sh outputs/submission.csv "First baseline submission"
# ---------------------------------------------------------

# Exit immediately on errors
set -e

COMPETITION=vic-vision-par-ordinateur-object-detection-in-baggage-using-classic

# --- Check arguments ---
if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <submission_file> \"<submission_message>\""
  exit 1
fi

FILE_PATH="$1"
MESSAGE="$2"

# --- Check if file exists ---
if [ ! -f "$FILE_PATH" ]; then
  echo "Error: File '$FILE_PATH' not found."
  exit 1
fi

# --- Submit to Kaggle ---
echo "🚀 Submitting '$FILE_PATH' to Kaggle competition: $COMPETITION"
uvx --from git+https://github.com/Kaggle/kaggle-api/ kaggle competitions submit -c "$COMPETITION" -f "$FILE_PATH" -m "$MESSAGE"

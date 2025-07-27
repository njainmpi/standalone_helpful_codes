#!/bin/bash

# Set the source and destination directories
SOURCE_DIR="/Users/njain/Desktop/Administration_Work"
DEST_DIR="/Users/njain/Desktop/test"

# Check if source directory exists
if [ ! -d "$SOURCE_DIR" ]; then
    echo "Source directory does not exist: $SOURCE_DIR"
    exit 1
fi

# Check if destination directory exists, create it if it doesn't
if [ ! -d "$DEST_DIR" ]; then
    echo "Destination directory does not exist. Creating: $DEST_DIR"
    mkdir -p "$DEST_DIR"
fi

# Run an infinite loop to monitor the source directory
while true; do
    # Check if there are any files or directories in the source directory
    if [ "$(ls -A "$SOURCE_DIR")" ]; then
        echo "Moving files and folders from $SOURCE_DIR to $DEST_DIR..."
        
        # Move files and folders recursively
        mv "$SOURCE_DIR"/* "$DEST_DIR"/
        
        echo "Move complete."
    else
        echo "No files to move. Waiting..."
    fi
    
    # Wait for a specified interval before checking again (e.g., 60 seconds)
    sleep 4800
done

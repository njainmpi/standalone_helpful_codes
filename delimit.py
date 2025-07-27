# Python script to reformat space-delimited data to another delimiter
import os
import sys
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import subprocess

# Specify input and output file paths
input_file = sys.argv[1]   # Replace with your input file name
output_file = sys.argv[2] # Replace with your output file name (use .csv for comma delimiter)

# Specify the desired output delimiter
delimiter = ','  # Change to '\t' for tab or another delimiter as needed

def reformat_space_delimited_file(input_path, output_path, delimiter):
    try:
        with open(input_path, 'r') as infile:
            lines = infile.readlines()

        processed_lines = []
        for line in lines:
            # Split line based on spaces (input is space-delimited)
            parts = line.strip().split()
            # Join the parts using the desired output delimiter
            processed_line = delimiter.join(parts)
            processed_lines.append(processed_line)
        
        # Write the processed lines to the output file
        with open(output_path, 'w') as outfile:
            outfile.write("\n".join(processed_lines))
        
        print(f"File reformatted successfully! Delimited file saved as '{output_path}'")
    except Exception as e:
        print(f"An error occurred: {e}")

# Call the function
reformat_space_delimited_file(input_file, output_file, delimiter)

#!/usr/bin/env python3
import os
import glob
import pandas as pd
import re

def extract_number(filename):
    """Extract the numeric part from a filename like 'ihdp_npci_1.csv'"""
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else 0

def main():
    # Set the input directory containing the CSV files.
    input_dir = os.path.join("data", "IHDP", "csv")
    
    # List all CSV files in the directory
    all_csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    
    # Exclude the concatenated file (if it exists) from processing.
    csv_files = [f for f in all_csv_files if os.path.basename(f) != "concatenated_ihdp.csv"]
    
    if not csv_files:
        print(f"No CSV files found in {input_dir} (after excluding the concatenated file).")
        return

    # Sort files by the numeric value in their names (e.g. 1,2,...,10)
    csv_files = sorted(csv_files, key=lambda f: extract_number(os.path.basename(f)))
    
    print("CSV files to be processed in order:")
    for f in csv_files:
        print(f" - {os.path.basename(f)}")
    
    # Define the expected columns (order matters)
    expected_headers = ["treatment", "y_factual", "y_cfactual", "mu0", "mu1"] + [f"x{i}" for i in range(1, 26)]
    expected_col_count = len(expected_headers)  # Should be 30

    # Read each CSV file with no header and check for validity.
    matching_files = []  # Files that have the expected number of columns.
    violating_files = [] # Files that do not match the expected scheme.
    valid_dataframes = []  # DataFrames from files matching the expected scheme.
    
    for file in csv_files:
        basename = os.path.basename(file)
        try:
            # Read without header, then drop any all-NaN columns (e.g., trailing empty columns)
            df = pd.read_csv(file, header=None)
            df = df.dropna(axis=1, how='all')
        except Exception as e:
            print(f"Error reading {basename}: {e}")
            violating_files.append(basename)
            continue

        if df.shape[1] == expected_col_count:
            # Assume the file is meant to have the expected scheme; assign headers.
            df.columns = expected_headers
            matching_files.append(basename)
            valid_dataframes.append(df)
        else:
            # Report missing or extra columns based on count.
            violating_files.append(basename)
            print(f"\nFile '{basename}' does NOT have the expected number of columns ({expected_col_count}).")
            print(f"  Columns found: {df.shape[1]}")
    
    # Report summary.
    print("\nSUMMARY:")
    if matching_files:
        print("Files matching the expected scheme (30 columns):")
        for f in matching_files:
            print(f" - {f}")
    else:
        print("No files match the expected scheme.")

    if violating_files:
        print("\nFiles NOT matching the expected scheme:")
        for f in violating_files:
            print(f" - {f}")

    # For concatenation, only use the files that match the expected scheme.
    if not valid_dataframes:
        print("\nNo valid CSV files to concatenate. Exiting.")
        return

    concatenated_df = pd.concat(valid_dataframes, ignore_index=True)
    
    # Define the output file path (save in the same directory)
    output_file = os.path.join(input_dir, "concatenated_ihdp.csv")
    
    # Save the concatenated dataframe to CSV without the index.
    concatenated_df.to_csv(output_file, index=False)
    print(f"\nConcatenated CSV saved to: {output_file}")

if __name__ == "__main__":
    main()

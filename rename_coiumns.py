#!/usr/bin/env python3
import os
import pandas as pd

def main():
    # Define file paths.
    original_file = "data/TWINS/processed_X.csv"
    shifted_file = "data/TWINS/processed_X_covariate_shifted.csv"
    output_file = "data/TWINS/processed_X_named.csv"  # New file with proper column names

    # Check that both files exist.
    if not os.path.exists(original_file):
        print(f"Error: {original_file} does not exist.")
        return
    if not os.path.exists(shifted_file):
        print(f"Error: {shifted_file} does not exist.")
        return

    # Load both datasets.
    df_original = pd.read_csv(original_file)
    df_shifted = pd.read_csv(shifted_file)

    # Verify that the number of columns is equal.
    if df_original.shape[1] != df_shifted.shape[1]:
        print("Error: The number of columns in the original and shifted files do not match.")
        print(f"Original columns: {df_original.shape[1]}, Shifted columns: {df_shifted.shape[1]}")
        return

    # Get column names from the shifted dataset.
    new_column_names = df_shifted.columns.tolist()

    # Assign the new column names to the original DataFrame.
    df_original.columns = new_column_names

    # Save the new DataFrame to a file.
    df_original.to_csv(output_file, index=False)
    print(f"New version saved to {output_file} with column names from the shifted set.")

if __name__ == "__main__":
    main()

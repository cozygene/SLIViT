import pandas as pd
import argparse
import os

def process_csv(csv_path, data_path, output_path):
    df = pd.read_csv(csv_path)

    normal_efs = df['EF'].between(0, 100)  #general sanity check
    assert normal_efs.all(), "Values in 'EF' must be between 0 and 100 (consider dropping invalid rows)"

    df['EFB'] = df['EF'].apply(lambda x: 1 if x < 50 else 0)  # Add the 'EFB' column
    assert (df['EF'] < 50).equals(df['EFB'] == 1), "EFB should be 1 if and only if EF < 0.5"

    df['path'] = df['FileName'].apply(lambda x: os.path.join(data_path, x))  # Add the 'path' column

    df.to_csv(output_path, index=False)
    print(f"Modified CSV has been saved to {output_path}")

    print(df.head())  # Show the first few rows for verification


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process CSV file and add columns.')
    parser.add_argument('--csv_path', default='./FileList.csv',
                        help='Path to the FileList.csv that comes with the dataset.')
    parser.add_argument('--data_path', required=True, help='Path to the local data directory')
    parser.add_argument('--out', required=True, help='Path to save the modified CSV file.')

    args = parser.parse_args()
    process_csv(args.csv_path, args.data_path, args.out)
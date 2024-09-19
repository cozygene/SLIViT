import pandas as pd
import argparse


def process_csv(input_csv, data_path, output_csv):
    df = pd.read_csv(input_csv)

    # Generate the 'pid' column by removing the '.jpeg' suffix from 'F_name'
    df['pid'] = df['F_name'].str.replace('.jpeg', '', regex=False)

    # Generate the 'path' column by prefixing the data path to the 'Path' column
    df['path'] = df['Path'].apply(lambda x: f'{data_path}{x}')

    output_df = df[['pid', 'path', 'Normal', 'DME', 'CNV', 'Drusen']]

    output_df.to_csv(output_csv, index=False)
    print(f"Processed CSV has been saved to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process the Kermany CSV file.')
    parser.add_argument('--input_csv', type=str, required=True,
                        help='Path to the original CSV file comes with the Kermany dataset.')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the data directory (where the train and test folders reside).')
    parser.add_argument('--output_csv', type=str, required=True, help='Path to save the processed CSV file.')

    args = parser.parse_args()

    process_csv(args.input_csv, args.data_path, args.output_csv)

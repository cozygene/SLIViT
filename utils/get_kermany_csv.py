import os
import pandas as pd
import argparse


def get_kermany_csv(data_path, output_csv):
    class_names = ['NORMAL', 'DRUSEN', 'CNV', 'DME']
    data = []
    # Traverse the directory structure
    for subfolder in os.listdir(data_path):
        subfolder_path = os.path.join(data_path, subfolder)

        if os.path.isdir(subfolder_path):
            for class_name in os.listdir(subfolder_path):
                class_path = os.path.join(subfolder_path, class_name)

                if os.path.isdir(class_path):
                    # Add the class name to the set of class names

                    for img_file in os.listdir(class_path):
                        if img_file.lower().endswith(('.jpeg')):  # Consider only jpeg files
                            # Construct the path and row information
                            file_path = os.path.join(class_path, img_file)
                            #     df['pid'] = df['F_name'].apply(lambda x: x.split('-')[1])
                            row = {
                                # Extract the 'pid' from filename.
                                # For example, CNV-1016042-100.jpeg -> 1016042
                                'pid': img_file.split('-')[1]
                            }

                            # Initialize class columns for this row
                            for cls in class_names:
                                row[cls] = int(class_name == cls)

                            row['path'] = file_path

                            # Add the row to the list
                            data.append(row)

    # Create a DataFrame from the list of rows
    df = pd.DataFrame(data)

    df.to_csv(output_csv, index=False)
    print(f"Processed CSV has been saved to {output_csv}")
    print(df.head())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create CSV from directory of images (should look like <path to unziped folder>/CellData/OCT')
    parser.add_argument('--data_path', required=True,
                        type=lambda x: x if os.path.exists(x) else None,
                        help='Path to the original Kermany\'s train and test folders.')
    parser.add_argument('--output_csv', type=str, default='./meta/kermany.csv',
                        help='Path to save the output CSV file.')

    args = parser.parse_args()

    get_kermany_csv(args.data_path, args.output_csv)

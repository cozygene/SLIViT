import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./', help='Path to the data')
parser.add_argument('--meta_csv', type=str, default='./Dsets/kermany_meta_mock.csv',
                    help='Path to the meta csv file')
parser.add_argument('--out_dir', type=str, default='./', help='Output directory to save the trained model')
parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID for training')
parser.add_argument('--b_size', type=int, default=4, help='Training batch size')
parser.add_argument('--n_cpu', type=int, default=1, help='# of cpus to use for loading data')
parser.add_argument('--n_epochs', type=int, default=20, help='# of training epochs')
parser.add_argument('-v', '--verbose', help='Increase output verbosity', action='store_true')

args = parser.parse_args()

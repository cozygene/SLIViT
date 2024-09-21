#TODO: prettify
import argparse

# general parameters
parser = argparse.ArgumentParser(prog='SLIViT',
                                 description='SLIViT: A deep-learning method for medical image diagnosis.')
parser.add_argument('--out_dir', type=str, default='.', help='Output directory to save the trained model')
parser.add_argument('--out_suffix', type=str, help='a subfolder name for hp tuning')
parser.add_argument('--meta_data', type=str, help='Path to meta csv file')
parser.add_argument('--test_csv', type=str, default=None,
                    help='Path to external test set csv file (use internal test set by default)')
# parser.add_argument('--split_file_path', type=str, help='Path to a meta csv file with a pre-defined split (train/val/test)')
parser.add_argument('--dataset', type=str, required=True, help='Task\'s dataset (2D or 3D)\n',
                    choices=['kermany', 'chestmnist', 'retinamnist', 'oct', 'ultrasound', 'mri', 'ct', 'custom'])
parser.add_argument('--mnist_mocks', type=int,
                    help='Number of mnist samples to use for this run. If None, all are used.')
parser.add_argument('--drop_default_suffix', action='store_true', help='Drop default suffix to the out_dir path')
parser.add_argument('--wandb_name', default=None, help='A name for the wandb project (if not provided, wandb will not be used)')

# advanced general parameters
parser.add_argument('--gpu_id', type=int, default=1, help='GPU ID for training')
parser.add_argument('--batch', type=int, default=16, help='Batch size')
parser.add_argument('--cpus', type=int, default=16, help='# of cpus to use for loading data')
parser.add_argument('--epochs', type=int, default=20, help='# of training epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--task', type=str, default='cls', help='cls (classification) or reg (regression)')
parser.add_argument('--seed', type=int, default=1, help='Set a random seed for reproducibility')
parser.add_argument('--data_dir', type=str, default='.',
                    help='Root path to the data')  # TODO: add kermany dir to meta csv
parser.add_argument('--split_ratio', type=lambda x: [float(i) for i in x.split(',')],
                    # TODO assert sum()==1 or make sure it's normalized
                    default=[0.85, 0.15, 0], help='train/val/test split. by default, data is used just for training (no test set).')
parser.add_argument('--min_delta', type=float, default=0.1,
                    help='minimum delta between the last monitor value and the best monitor value for early stopping')
parser.add_argument('--patience', type=int, default=5, help='patience for early stopping')
parser.add_argument('--fine_tune', action='store_true', help='learner.fine_tune instead of learner.fit')

# pretraining
parser.add_argument('--label2d', type=lambda x: x.split(','),
                    help='Comma separated list of labels to use as targets in the pre-training')  # TODO: automate this and/or merge with label3d

# fine_tuning and evaluation parameters
parser.add_argument('--fe_path', type=str, help='Path to a pre-rained feature extractor',
                    default='/scratch/avram/projects/hadassah/CRORA_imputation/models/kermany/convnext_tiny_feature_extractor.pth')
parser.add_argument('--label3d', type=str, default=None, help='goal of the 3D task')  # TODO: merge with pathologies
parser.add_argument('--checkpoint', type=str, default='./slivit_ct',  # TODO: remove this default
                    help='path to a finetuned SLIViT model')
parser.add_argument('--fe_classes', type=int, default=4,
                    help='# of classes the feature extractor was pretrained on (e.g., Kermany: 4, ChestMNIST: 14)')
parser.add_argument('--vit_depth', type=int, default=5, help='Feature Integrator depth')  # TODO: change dim to fi_depth
parser.add_argument('--vit_dim', type=int, default=256, help='Feature Integrator dim')  # TODO: change dim to fi_width
parser.add_argument('--mlp_dim', type=int, default=512, help='FC layer dimension')
parser.add_argument('--slices', type=int, default=28, help='# of volume\'s 2D frames to use for to use')
parser.add_argument('--heads', type=int, default=20, help='# of heads of the multi-head attention')
parser.add_argument('--dropout', type=float, default=0)
parser.add_argument('--emb_dropout', type=float, default=0)  # TODO: what's that?
parser.add_argument('--sparsing_method', type=str, default='eq',
                    help='Downsampling method for 3D data in case of different slice numbers',
                    choices=['eq', 'mid', 'custom'])
parser.add_argument('--split_col', type=str, default='split', help='Column name in the meta csv file for the split')
parser.add_argument('--pid_col', type=str, default='pid', help='Patient ID column name in the meta csv')
parser.add_argument('--path_col', type=str, default='path', help='Volume paths column name in the meta csv')

args = parser.parse_args()
# print(args)


# import argparse
# import sys
#
# def setup_parser():
#     # Main parser
#     parser = argparse.ArgumentParser(description="SLIViT: A deep-learning methods for medical image diagnosis.")
#
#     # Setup subparsers for each running mode
#     subparsers = parser.add_subparsers(dest='mode', required=True, help='Running modes')
#     setup_backbone(subparsers)
#     setup_slivit(subparsers)
#
#     return parser
#
#
# def setup_backbone(subparsers):
#     parser_pretrain = subparsers.add_parser('pretrain', help='Pretraining setup')
#     # Subcommand specific arguments
#     parser_pretrain.add_argument('--dataset', type=str, default='kermany', help='kermany,chestmnist,custom')
#     parser_pretrain.add_argument('--pathologies', type=str, default='CNV,Drusen,DME,Normal',
#                                  help='Comma Spreaded List of Labels to predict for Pre-training')
#     parser_pretrain.add_argument('--min_delta', type=float, default=0.1,
#                                  help='minimum delta between the last monitor value and the best monitor value for early stopping')
#     parser_pretrain.add_argument('--patience', type=float, default=20, help='patience for early stopping')
#
#
# def setup_slivit(subparsers):
#     parser = subparsers.add_parser('slivit', help='Training setup')
#     # Subcommand specific arguments
#
#     parser.add_argument('--bbpath', type=str, help='Path to pre-rained Convnext Backbone',
#                               default='/scratch/avram/projects/hadassah/CRORA_imputation/convnext_bb_kermany.pth')
#     parser.add_argument('--checkpoint', type=str, default='./slivit_ct',
#                               help='Path to a fin-tuned SLIViT model')
#     parser.add_argument('--nObb_feat', type=int, default=4, help='# of features backbone extracts')
#     parser.add_argument('--depth', type=int, default=5, help='ViT depth')
#     parser.add_argument('--dim', type=int, default=256, help='ViT dim')
#     parser.add_argument('--mlp_dim', type=int, default=512, help='fc dim')
#     parser.add_argument('--nslc', type=int, default=28, help='# of slices to use for 3D Fine-tuning')
#     parser.add_argument('--heads', type=int, default=20, help='# of heads for multihead attention')
#     parser.add_argument('--dropout', type=float, default=0)
#     parser.add_argument('--emb_dropout', type=float, default=0)
#     parser.add_argument('--pathology', type=str, help='Label of the 3D learning task')
#     parser.add_argument('--sparsing_method', type=str, default='eq',
#                               help='Downsampling method for 3D data in case of different slice numbers',
#                               choices=['eq', 'mid', 'custom'])
#
#     parser.add_argument('--min_delta', type=float, default=0.1,
#                               help='minimum delta between the last monitor value and the best monitor value for early stopping')
#     parser.add_argument('--patience', type=float, default=20, help='patience for early stopping')
#
#
# def parse_args():
#     parser = setup_parser()
#
#     # First parse known args to identify the mode
#     args, remaining_argv = parser.parse_known_args()
#
#     # Define common arguments that can appear after the subcommand
#     parser = argparse.ArgumentParser(add_help=False)
#     common_parser.add_argument('--data_dir', type=str, default='.', help='Root path to the data')
#     common_parser.add_argument('--meta_data', type=str, default='no_meta', help='Path to meta csv file')
#     common_parser.add_argument('--out_dir', type=str, default='.', help='Output directory to save the trained model')
#     common_parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID for training')
#     common_parser.add_argument('--b_size', type=int, default=16, help='Batch size for training')
#     common_parser.add_argument('--n_cpu', type=int, default=16, help='# of cpus to use for loading data')
#     common_parser.add_argument('--n_epochs', type=int, default=20, help='# of training epochs')
#     common_parser.add_argument('--task', type=str, default='cls', help='cls (classification) or reg (regression)')
#     common_parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
#     common_parser.add_argument('--split_ratio',
#                                # TODO assert sum()==1,
#                                default=(.7, .15, .15), help='train/val/test split')
#
#     # Re-parse all arguments with the common options included
#     full_parser = setup_parser()
#     full_parser.add_argument('--data_dir', type=str, default='.', help='Root path to the data')
#     full_parser.add_argument('--meta_data', type=str, default='no_meta', help='Path to meta csv file')
#     full_parser.add_argument('--out_dir', type=str, default='.', help='Output directory to save the trained model')
#     full_parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID for training')
#     full_parser.add_argument('--b_size', type=int, default=16, help='Batch size for training')
#     full_parser.add_argument('--n_cpu', type=int, default=16, help='# of cpus to use for loading data')
#     full_parser.add_argument('--n_epochs', type=int, default=20, help='# of training epochs')
#     full_parser.add_argument('--task', type=str, default='cls', help='cls (classification) or reg (regression)')
#     full_parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
#     full_parser.add_argument('--split_ratio',
#                                # TODO assert sum()==1,
#                                default=(.7, .15, .15), help='train/val/test split')
#
#     # The final complete args parsing
#     args = full_parser.parse_args(remaining_argv + sys.argv[1:])
#     return args
#
#
# if __name__ == "__main__":
#     args = parse_args()
#     print(args)
#
# args = parse_args()

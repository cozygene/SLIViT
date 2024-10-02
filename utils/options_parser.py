import argparse

# Initialize parser
parser = argparse.ArgumentParser(
    prog='SLIViT',
    description='SLIViT: A deep-learning method for medical image diagnosis.'
)

# General parameters
parser.add_argument('--out_dir', type=str, required=True, help='Output directory to save the trained model.')
parser.add_argument('--out_suffix', type=str, help='Subfolder name for hyperparameter tuning.')
parser.add_argument('--meta', type=str, help='Path to the metadata CSV file.')
parser.add_argument('--test_meta', type=str, default=None,
                    help='Path to external test set CSV file (uses internal test set by default).')
parser.add_argument('--label', type=lambda x: x.split(','),
                    help='Goal of the learning task (comma-separated if more than one).')
parser.add_argument('--dataset_name', type=str, required=True,
                    choices=['oct2d', 'xray2d', 'custom2d', 'oct3d', 'us3d', 'mri3d', 'ct3d', 'custom3d'])
parser.add_argument('--drop_default_suffix', action='store_true',
                    help='Drop the default suffix from the output directory path.')
parser.add_argument('--wandb_name', type=str, default=None,
                    help='Weights & Biases project name (if not provided, WandB will not be used).')

# Advanced general parameters
parser.add_argument('--gpu_id', type=str, default='0', help='GPU ID for training (default: 0).')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training.')
parser.add_argument('--cpus', type=int, default=16, help='Number of CPU workers for data loading.')
parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs.')
parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate for training.')
parser.add_argument('--task', type=str, default='cls', help='Task type: "cls" (classification) or "reg" (regression).')
parser.add_argument('--seed', type=int, default=1, help='Set random seed for reproducibility.')
parser.add_argument('--medmnist_mocks', type=int,
                    help='Number of MedMNIST samples to use for this run (default: all samples).')
parser.add_argument('--medmnist_root', type=str, default='./data',
                    help='Root directory for MedMNIST dataset downloads.')
parser.add_argument('--split_ratio', type=lambda x: [float(i) for i in x.split(',')], default=[0.85, 0.15, 0],
                    help='Train/Val/Test split ratio (comma-separated).')
parser.add_argument('--min_delta', type=float, default=0, help='Minimum delta for early stopping.')
parser.add_argument('--patience', type=int, default=5,
                    help='Number of epochs to wait for improvement before early stopping.')
parser.add_argument('--finetune', action='store_true', help='Use learner.fine_tune() instead of learner.fit().')

# Fine-tuning and evaluation parameters
parser.add_argument('--fe_path', type=str,
                    default='./checkpoints/kermany/feature_extractor.pth',
                    help='Path to the pretrained feature extractor.')
parser.add_argument('--fe_classes', type=int, default=4,
                    help='Number of classes in the pretrained feature extractor '
                         '(for oct2d it\'s 4; for xray2d it\'s 14).')
parser.add_argument('--checkpoint', type=str, default=None, help='Path to the finetuned SLIViT model.')
parser.add_argument('--ignore_options_file', action='store_true',
                    help='Forcing to use hps provided by the user (even if there is an options file).')
parser.add_argument('--vit_depth', type=int, default=5, help='Depth of the Vision Transformer (ViT).')
parser.add_argument('--vit_dim', type=int, default=256, help='Dimension of the Vision Transformer (ViT).')
parser.add_argument('--mlp_dim', type=int, default=512, help='Dimension of the MLP layer.')
parser.add_argument('--slices', type=int, default=28,
                    help='Number of 2D slices to use from the 3D volume.')  # TODO: consider changing the default to 32
parser.add_argument('--heads', type=int, default=20, help='Number of heads in the multi-head attention mechanism.')
parser.add_argument('--dropout', type=float, default=0, help='Dropout rate for training.')
parser.add_argument('--emb_dropout', type=float, default=0, help='Dropout rate for embeddings.')
parser.add_argument('--img_suffix', type=str, default='tiff', help='File suffix to filter images (e.g., tiff, png).')
parser.add_argument('--sparsing_method', type=str, default='eq', choices=['eq', 'mid', 'custom'],
                    help='Method for standardizing 3D data when there are different slice counts.')
parser.add_argument('--split_col', type=str, default='split',
                    help='Column name in the metadata CSV for the train/val/test split.')
parser.add_argument('--pid_col', type=str, default='pid', help='Patient ID column name in the metadata CSV.')
parser.add_argument('--path_col', type=str, default='path', help='Volume paths column name in the metadata CSV.')

args = parser.parse_args()

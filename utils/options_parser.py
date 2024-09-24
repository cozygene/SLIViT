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
parser.add_argument('--dataset', type=str, required=True,
                    # choices=['kermany', 'chestmnist', 'retinamnist', 'oct', 'ultrasound', 'mri', 'ct', 'custom'],
                    help='Task\'s dataset (2D or 3D)\n')
parser.add_argument('--mnist_mocks', type=int,
                    help='Number of mnist samples to use for this run. If None, all are used.')
parser.add_argument('--drop_default_suffix', action='store_true', help='Drop default suffix to the out_dir path')
parser.add_argument('--wandb_name', default=None, help='A name for the wandb project (if not provided, wandb will not be used)')

# advanced general parameters
parser.add_argument('--gpu_id', type=str, default='0', help='GPU ID for training')
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
parser.add_argument('--finetune', action='store_true', help='learner.fine_tune instead of learner.fit')

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
parser.add_argument('--emb_dropout', type=float, default=0)
parser.add_argument('--img_suffix', type=str, default='tiff', help='Images are filtered by this suffix (other files are ignored)')
parser.add_argument('--sparsing_method', type=str, default='eq',
                    help='Downsampling method for 3D data in case of different slice numbers',
                    choices=['eq', 'mid', 'custom'])
parser.add_argument('--split_col', type=str, default='split', help='Column name in the meta csv file for the split')
parser.add_argument('--pid_col', type=str, default='pid', help='Patient ID column name in the meta csv')
parser.add_argument('--path_col', type=str, default='path', help='Volume paths column name in the meta csv')

args = parser.parse_args()

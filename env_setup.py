import os

from utils.options_parser import args
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

from fastai.callback.progress import CSVLogger
from utils.slivit_auxiliaries import save_options, logger, get_script_name, wrap_up, set_seed, Subset

out_dir = args.out_dir.rstrip('/')
if not args.drop_default_suffix:
    # by default, the csv file name (or dataset name in case of mnist) is added to the output directory
    if args.meta_csv is not None:
        # example args.meta_csv:
        # /meta_file_folder_path/ultrasound.csv
        csv_file_name = os.path.splitext(args.meta_csv.split("/")[-1])[0]  # remove extension
        out_dir = f'{out_dir}/{csv_file_name}'
    else:
        out_dir = f'{out_dir}/{"mock_" if args.mnist_mocks else ""}{args.dataset}'

if args.out_suffix is not None:
    # subfolders for hp search
    out_dir += f'/{args.out_suffix}'

logger.info(f'\nOutput direcory is\n{out_dir}\n')
os.makedirs(out_dir, exist_ok=True)

from fastai.vision.all import *

options_file = f'{out_dir}/options_{get_script_name()}.txt'
save_options(options_file, args)
logger.info(f'Running configuration is saved at:\n{options_file}\n')

if args.seed is not None:
    set_seed(args.seed)
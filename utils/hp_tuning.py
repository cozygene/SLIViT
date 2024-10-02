import argparse
import os
import random
import subprocess
import time
import pandas as pd


def setup_parser():
    parser = argparse.ArgumentParser(description='Script to process configurations based on a CSV file.')

    # Labels to learn
    parser.add_argument('--labels', type=lambda x: x.split(','), required=True,
                        help='Comma-separated list of labels to process.')

    # Path to the CSV file
    parser.add_argument('--train_csv_path', type=str, required=True, help='Path to the train meta file.')

    # Path to the test set
    parser.add_argument('--test_meta_path', type=str, required=True, help='Path to the test meta file.')

    # Number of configurations to process
    parser.add_argument('--num_configs', type=int, required=True, help='Number of configurations to process.')

    # Feature extractor path
    parser.add_argument('--fe_path', type=str, default='./checkpoints/kermany/feature_extractor.pth',
                        help='Path to the pretrained feature extractor.')
    # GPU ID to use
    parser.add_argument('--gpu_id', type=str, default='3,1', help='GPU ID for training')

    # Mock processing mode
    parser.add_argument('--mock', action='store_true', help='Run with mock data.')

    # Output path for saving processing results or logs
    parser.add_argument('--out', type=str, required=True, help='Path to save the output of the processing.')

    args = parser.parse_args()

    if args.mock:
        args.train_csv_path = '/mock_'.join(os.path.split(args.train_csv_path))
        args.test_meta_path = '/mock_'.join(os.path.split(args.test_meta_path))
        args.out += '/mock'

    # Print all parsed arguments
    print("Parsed Arguments:")
    print(f"Labels: {args.labels}")
    print(f"Train data: {args.test_meta_path}")
    print(f"Test data: {args.train_csv_path}")
    print(f"Number of Configurations: {args.num_configs}")
    print(f"GPU ID: {args.gpu_id}")
    print(f"Mock Mode: {'Enabled' if args.mock else 'Disabled'}")
    print(f"Output Path: {args.out}")

    return args


args = setup_parser()
hyper_params = {
    'batch': [4],  # 1],
    'lr': [1e-4, 5e-5],  # 1e-3],
    'vit_depth': [1, 5, 9],
    'vit_dim': [64, 128, 256],
    'mlp_dim': [64],
    'heads': [10, 20, 30, 40],
    'dropout': [0.1, 0.05],
    'emb_dropout': [0.1],
    'sparsing_method': ['eq', 'mid'],
    'epochs': [1] if args.mock else [5, 10, 20],
    'slices': [19, 49],
    'finetune': [False, True]
}


def get_configurations(num_configs):
    random.seed(42)

    configs = tuple()  # to track used configurations

    # Keep generating until enough configurations are generated
    while len(configs) < num_configs:
        config = tuple()
        for hp in hyper_params:
            value = random.choice(hyper_params[hp])
            config += ((hp, value),)

        if config not in configs:
            configs += (config,)
        # else:
        #     print("Duplicate configuration found. Skipping.")

    return configs


def run_commands(fe_path, csv_path, test_path, num_configs, gpu, labels, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    configs = get_configurations(num_configs)

    configurations_path = f'{out_dir}/configurations.csv'
    pd.DataFrame([dict(config) for config in configs]).to_csv(configurations_path, index=False,
                                                              mode='a' if os.path.exists(configurations_path) else 'w',
                                                              header=not os.path.exists(configurations_path))
    pd.read_csv(configurations_path).drop_duplicates(keep='first').to_csv(configurations_path, index=False)
    f'cat {configurations_path} |sort|uniq -c | grep "2 "|wc'  # should return 0 now

    for config in configs:
        for label in labels:
            # print(config)
            finetune = config[-1][-1]
            suffix = '-'.join(f'{item[0]}_{item[1]}' for item in config if item[0] != 'finetune')
            suffix += f'-finetune' if finetune else '-fit'
            command = f"/scratch/avram/envs/slivit/bin/python /scratch/avram/projects/SLIViT/finetune.py "
            command += f"--seed 1 --dataset oct3d "
            command += f'--fe_classes 4 '
            command += f'--meta {csv_path} '
            command += f'--fe_path {fe_path} '
            command += f'--test_meta {test_path} '
            command += f"--label {label} --gpu_id {gpu} --cpus 40 "
            command += f"--drop_default_suffix "
            command += f"--split_ratio .8,.2,.0 "
            command += f"--wandb_name houston_{label} "
            command += f"--out_dir {out_dir}/{label}/{suffix} "
            command += ' '.join(f'--{item[0]} {item[1]}' for item in config if item[0] != 'finetune')
            if finetune:
                command += f' --finetune'
            done_file_path = f'{out_dir}/{label}/{suffix}/done_finetune'  # TODO: change to {get_script_name()}

            try:
                if os.path.getsize(done_file_path) == 0:
                    print('Configuration already trained successfully. Skipping.')
                    continue  # skip configuration
                else:
                    # unsuccessful training
                    os.remove(done_file_path)
            except FileNotFoundError:
                # never trained
                pass

            print(f"Running command:\n{command}\n")
            subprocess.run(command, shell=True)

            # Check for the completion file
            while not os.path.exists(done_file_path):
                time.sleep(10)


run_commands(args.fe_path, args.train_csv_path, args.test_meta_path,
             args.num_configs, args.gpu_id, args.labels, args.out)

print(f"Ran {args.num_configs} configurations.")

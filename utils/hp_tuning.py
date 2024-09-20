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
    parser.add_argument('--test_csv_path', type=str, required=True, help='Path to the test meta file.')

    # Number of configurations to process
    parser.add_argument('--num_configs', type=int, required=True, help='Number of configurations to process.')

    # GPU ID to use
    parser.add_argument('--gpu_id', type=int, required=True, help='GPU ID to use for processing.')

    # Mock processing mode
    parser.add_argument('--mock', action='store_true', help='Run with mock data.')

    # Output path for saving processing results or logs
    parser.add_argument('--out', type=str, required=True, help='Path to save the output of the processing.')

    args = parser.parse_args()

    if args.mock:
        args.train_csv_path = '/mock_'.join(os.path.split(args.train_csv_path))
        args.test_csv_path = '/mock_'.join(os.path.split(args.test_csv_path))
        args.out += '/mock'

    # Print all parsed arguments
    print("Parsed Arguments:")
    print(f"Labels: {args.labels}")
    print(f"Train data: {args.test_csv_path}")
    print(f"Test data: {args.train_csv_path}")
    print(f"Number of Configurations: {args.num_configs}")
    print(f"GPU ID: {args.gpu_id}")
    print(f"Mock Mode: {'Enabled' if args.mock else 'Disabled'}")
    print(f"Output Path: {args.out}")

    return args


args = setup_parser()
hyper_params = {
    'batch': [4],  # 1],
    'lr': [5e-3, 1e-4, 5e-4],  # 1e-3],
    'vit_depth': [1, 5, 9],
    'vit_dim': [32, 64, 128],
    'mlp_dim': [64],
    'heads': [19, 49],
    'dropout': [0.1, 0.2],
    'emb_dropout': [0, 0.1],
    'sparsing_method': ['eq', 'mid'],
    'epochs': [1] if args.mock else [5, 10, 20],
    'slices': [19, 49],
    'fine_tune': [False, True]
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
        else:
            print("Duplicate configuration found. Skipping.")

    return configs


def run_commands(csv_path, test_path, num_configs, gpu, labels, out_dir):
    # meta_csv='/scratch/avram/projects/hadassah/CRORA_imputation/meta/hp_tuning_crora_manual_labels.csv'):
    # out_dir = f'/scratch/avram/projects/hadassah/CRORA_imputation/models/{mock if mock else ""}hp_tuning_crora'

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
            tuning = 'fine_tune' if config[-1] else 'fit'
            suffix = '-'.join(f'{item[0]}_{item[1]}' for item in config if item[0] != 'fine_tune') + f'-{tuning}'
            command = f"/scratch/avram/envs/slivit/bin/python /scratch/avram/projects/SLIViT/finetune.py "
            command += f"--seed 1 --dataset oct "
            command += f'--meta_csv {csv_path} '
            command += f'--test_csv {test_path} '
            command += f"--label3d {label} --gpu_id {gpu} --cpus 40 "
            command += f"--drop_default_suffix "
            command += f"--split_ratio .8,.2,.0 "
            command += f"--wandb_name houston_{label} "
            command += f"--out_dir {out_dir}/{label}/{suffix} "
            command += ' '.join(
                f'--{item[0]} {item[1]}' for item in config if item[0] != 'fine_tune') + f' --{tuning}' + '\n'
            done_file_path = f'{out_dir}/{label}/{suffix}/done_finetune' # TODO: change to {get_script_name()}

            if os.path.exists(done_file_path):
                print('Configuration already trained. Skipping.')
                # break
                continue

            if not os.path.exists(done_file_path):
                print(f"Running command:\n{command}")
                subprocess.run(command, shell=True)
            else:
                print(f'Configuration already trained. Skipping...')

            # Check for the completion file
            while not os.path.exists(done_file_path):
                time.sleep(10)


run_commands(args.train_csv_path, args.test_csv_path, args.num_configs,
             args.gpu_id,args.labels, args.out)
print(f"Ran {args.num_configs} configurations.")

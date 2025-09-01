import os
import gc
import random
import torch
import logging
import argparse
import numpy as np
import pandas as pd

from models.AMIO import AMIO
from trains.multiTask.SAFE_trainer import SAFE_trainer
from data.load_data import MMDataLoader
from config.config_regression import ConfigRegression

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def run(args):
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    args.model_save_path = os.path.join(args.model_save_dir,
                                        f'{args.modelName}-{args.datasetName}-{args.train_mode}-seed-{args.seed}.pth')

    if not args.gpu_ids:
        if torch.cuda.is_available():
            args.gpu_ids.append(0)

    device = torch.device(f'cuda:{args.gpu_ids[0]}' if torch.cuda.is_available() else 'cpu')
    args.device = device

    dataloader = MMDataLoader(args)
    model = AMIO(args).to(device)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f'The model has {count_parameters(model)} trainable parameters')

    trainer = SAFE_trainer(model, args)

    best_epoch_results = trainer.do_train(dataloader)

    logger.info(f"Loading best model from {args.model_save_path} for final test...")
    assert os.path.exists(args.model_save_path)
    model.load_state_dict(torch.load(args.model_save_path))
    model.to(device)

    test_dataloader = dataloader['valid'] if args.get('tune_mode', False) else dataloader['test']
    mode = "VALID" if args.get('tune_mode', False) else "TEST"

    test_results = trainer.do_test(test_dataloader, mode=mode)

    del model
    torch.cuda.empty_cache()
    gc.collect()

    return test_results, best_epoch_results


def run_normal(args):
    args.res_save_dir = os.path.join(args.res_save_dir, 'normals')
    if not os.path.exists(args.res_save_dir):
        os.makedirs(args.res_save_dir)

    seeds = args.seeds

    current_run_results = []

    for i, seed in enumerate(seeds):
        current_args = argparse.Namespace(**vars(args))
        if current_args.train_mode == "regression":
            config = ConfigRegression(current_args)

        run_args = config.get_config()
        setup_seed(seed)
        run_args.seed = seed
        logger.info(f'Start running {run_args.modelName} on seed {seed}...')

        test_results, best_epoch_results = run(run_args)

        current_run_results.append(test_results)

        criterions = list(test_results.keys())
        save_path = os.path.join(run_args.res_save_dir,
                                 f'{run_args.modelName}-{run_args.datasetName}-{run_args.train_mode}.csv')

        if os.path.exists(save_path):
            df = pd.read_csv(save_path)
        else:
            df = pd.DataFrame(columns=["Model", "Seed"] + criterions)

        res = [run_args.modelName, seed]
        for c in criterions:
            res.append(round(test_results[c], 4))
        df.loc[len(df)] = res

        df.to_csv(save_path, index=False)
        logger.info(f'Results for seed {seed} have been saved to {save_path}')

        log_dir = 'results/logs'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_file_path = os.path.join(log_dir, f'{run_args.modelName}-{run_args.datasetName}-best-epochs.txt')

        with open(log_file_path, 'a') as f:
            header = "Best_Epoch,MAE,Non0_acc_2,Mult_acc_7,Corr,Non0_F1_score,Loss,Seed\n"
            log_keys = ['epoch', 'MAE', 'Non0_acc_2', 'Mult_acc_7', 'Corr', 'Non0_F1_score', 'Loss']

            if os.path.getsize(log_file_path) == 0:
                f.write(header)

            log_parts = []
            for key in log_keys:
                value = best_epoch_results.get(key, 'N/A')
                if isinstance(value, float):
                    log_parts.append(f"{value:.4f}")
                else:
                    log_parts.append(str(value))

            log_parts.append(str(seed))

            log_line = ",".join(log_parts) + "\n"
            f.write(log_line)

    if current_run_results:
        df_final = pd.read_csv(save_path)

        avg_res = [args.modelName, 'avg']
        for c in criterions:
            values = [result[c] for result in current_run_results]
            avg_value = np.mean(values)
            avg_res.append(round(avg_value, 4))

        df_final.loc[len(df_final)] = avg_res
        df_final.to_csv(save_path, index=False)

        logger.info(f'Average results for current run ({len(seeds)} seeds) have been saved to {save_path}')
        logger.info(f'Seeds used in this run: {seeds}')

        avg_summary = "Current run average results: "
        for i, c in enumerate(criterions):
            avg_summary += f"{c}: {avg_res[i + 2]:.4f}, "
        logger.info(avg_summary.rstrip(', '))


def set_log(args):
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file_path = os.path.join(log_dir, f'{args.modelName}-{args.datasetName}.log')

    logger = logging.getLogger('MSA')
    logger.setLevel(logging.DEBUG)

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    fh = logging.FileHandler(log_file_path, mode='w')
    fh.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(file_formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    ch.setFormatter(console_formatter)
    logger.addHandler(ch)

    return logger

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_mode', type=str, default="regression", help='The task type (e.g., regression)')
    parser.add_argument('--modelName', type=str, default='safe', help='The name of the model to run (e.g., safe)')
    parser.add_argument('--datasetName', type=str, default='mosi', help='The name of the dataset (e.g., mosi/mosei/sims)')
    parser.add_argument('--num_workers', type=int, default=4, help='The number of worker threads for data loading')
    parser.add_argument('--model_save_dir', type=str, default='results/models', help='The directory to save the models')
    parser.add_argument('--res_save_dir', type=str, default='results/results', help='The directory to save the results')
    parser.add_argument('--gpu_ids', nargs='+', type=int, default=[0], help='The GPU IDs to use')
    parser.add_argument('--seeds', nargs='+', type=int, default=[1111], help='A list of random seeds for the experiments')

    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()

    my_custom_seeds = [1100]
    args.seeds = my_custom_seeds
    logger = set_log(args)
    run_normal(args)

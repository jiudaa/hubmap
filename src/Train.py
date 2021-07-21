import sys
import argparse
import yaml
import pathlib
import os
import torch
from torch.utils.data import DataLoader
torch.backends.cudnn.benchmark = True
from transform import get_train_transforms, get_valid_transforms
import dataset
import loss
import metrics
import trainer
from network.model import build_model
import pandas as pd
from util import util
import sys
import numpy as np

sys.path.append('/data/p303872/HUBMAP/code/src/network')

def main(args):
    path_to_config = pathlib.Path(args.path)
    with open(path_to_config) as f:
        config = yaml.safe_load(f)

    # read config:
    seed = config['seed']
    util.seed_everything(seed)
    FOLD_LIST = config['FOLD_LIST']
    val_patient_numbers_list=config['val_patient_numbers_list']
    data_path = pathlib.Path(config['data_path'])
    csv_path = pathlib.Path(config['csv_path'])
    output_dir = pathlib.Path(config['output_dir'])
    
    train_batch_size = int(config['train_batch_size'])
    val_batch_size = int(config['val_batch_size'])
    num_workers = int(config['num_workers'])
    lr = float(config['lr'])
    n_epochs = int(config['n_epochs'])
    n_cls = int(config['n_cls'])
    T_0 = int(config['T_0'])
    eta_min = float(config['eta_min'])
    #model parameter
    baseline = config['baseline']
    resolution = config['resolution']
#     loss_name = config['loss_name']
    deepsupervision = config['deepsupervision']
    clfhead=config['clfhead']
    clf_threshold = config['clf_threshold']
    
    dice_threshold=config['dice_threshold']
    small_mask_threshold = config['small_mask_threshold']
    multiplier_bin=config['multiplier_bin']
    pretrain_path_list = config['pretrain_path_list']
    early_stopping = config['early_stopping']
    scheduler_step_per_epoch = config['scheduler_step_per_epoch']
    binned_max=config['binned_max']
    os.makedirs(output_dir,exist_ok=True)

    # train and val data paths:
    data_df = pd.read_csv(csv_path)
#     print(data_df)
    train_files_name, val_files_name= util.get_train_val_paths(data_df,val_patient_numbers_list,
                                                               multiplier_bin,binned_max,seed,output_dir)
    # train and val data transforms:
    for fold in FOLD_LIST:
        print(f'training on fold{fold}')
        os.makedirs(str(output_dir)+f'fold{fold}',exist_ok=True)

    # datasets:
        train_set = dataset.HUBMAPDataset(train_files_name[fold],get_train_transforms)
        val_set = dataset.HUBMAPDataset(val_files_name[fold],get_valid_transforms)

        # dataloaders:
        train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_set, batch_size=val_batch_size, shuffle=False, num_workers=num_workers)

        dataloaders = {
            'train': train_loader,
            'val': val_loader
        }
        model=build_model(model_name=baseline,
                            resolution=resolution, 
                            deepsupervision=deepsupervision, 
                            clfhead=clfhead,
                            clf_threshold=clf_threshold,
                            load_weights=True)
        if pretrain_path_list is not None:
            model.load_state_dict(torch.load(pretrain_path_list[fold]))
            
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, eta_min=eta_min)

        trainer_ = trainer.ModelTrainer(
            output_dir=output_dir,
            fold=fold,
            model=model,
            dataloaders=dataloaders,
            criterion=True,
            optimizer=optimizer,
            deepsupervision=deepsupervision,
            clfhead=clfhead,
            clf_threshold=clf_threshold,
            dice_threshold=dice_threshold,
            small_mask_threshold=small_mask_threshold,
            early_stopping=early_stopping,
            metric=None,
            scheduler=scheduler,
            num_epochs=n_epochs,
            parallel=False
        )

        trainer_.train_model()
        trainer_.save_results(path_to_dir=str(output_dir)+f'fold{fold}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model Training Script')
    parser.add_argument("-p", "--path", type=str, required=True, help="path to the config file")
    args = parser.parse_args()
    main(args)
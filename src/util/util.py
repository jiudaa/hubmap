import os
import pandas as pd
import torch
import random
import numpy as np
import pickle

def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
def get_train_val_paths(data_df,val_patient_numbers_list,multiplier_bin,binned_max,seed,output_path):
    data_df = data_df[data_df['std_img']>10].reset_index(drop=True)
    data_df['binned'] = np.round(data_df['ratio_masked_area'] * multiplier_bin).astype(int)
    data_df['is_masked'] = data_df['binned']>0
    
    # train
    trn_idxs_list, val_idxs_list = get_fold_idxs_list(data_df, val_patient_numbers_list)
    with open(os.path.join(output_path,f'trn_idxs_list_seed{seed}.pickle'), 'wb') as f:
        pickle.dump(trn_idxs_list, f)
    with open(os.path.join(output_path,f'val_idxs_list_seed{seed}.pickle'), 'wb') as f:
        pickle.dump(val_idxs_list, f)
    return trn_idxs_list,val_idxs_list

        
        
def get_fold_idxs_list(trn_df, val_patient_numbers_list,sample=True):
    trn_idxs_list = []
    val_idxs_list = []
    for fold in range(len(val_patient_numbers_list)):
        trn_idxs = trn_df[~trn_df['patient_number'].isin(val_patient_numbers_list[fold])].filename.tolist()
        n_sample = trn_df['is_masked'].value_counts().min()
        trn_df_0 = trn_df[trn_df['is_masked']==False].sample(n_sample, replace=True)
        trn_df_1 = trn_df[trn_df['is_masked']==True].sample(n_sample, replace=True)
        n_bin = int(trn_df_1['binned'].value_counts().mean())
        trn_df_list = []
        for bin_size in trn_df_1['binned'].unique():
            trn_df_list.append(trn_df_1[trn_df_1['binned']==bin_size].sample(n_bin, replace=True))
        trn_df_1 = pd.concat(trn_df_list, axis=0)
        trn_df_balanced = pd.concat([trn_df_1, trn_df_0], axis=0).reset_index(drop=True)

        trn_idxs_list.append(np.array(trn_idxs))
        val_idxs = trn_df[trn_df['patient_number'].isin(val_patient_numbers_list[fold])].filename.tolist()
        val_idxs_list.append(np.array(val_idxs))
    return trn_idxs_list, val_idxs_list
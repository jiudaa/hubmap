import os
import pathlib
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import codecs
import datetime
from torch import nn
from loss import criterion_lovasz_hinge_non_empty,lovasz_hinge,DiceLoss,criterion_dice_non_empty
from metrics import dice_sum_2,dice_sum
import gc
class ModelTrainer:
    def __init__(self,output_dir,fold,model,dataloaders,criterion,optimizer,deepsupervision,clfhead,clf_threshold,dice_threshold,
                 small_mask_threshold,early_stopping=True,metric=None,mode='max',scheduler=None,num_epochs=25,parallel=False,
                 device='cuda:0',save_last_model=False,scheduler_step_per_epoch=True):
        self.output_dir = output_dir
        self.fold = fold
        self.model = model
        self.dataloaders = dataloaders
        self.criterion = criterion
        self.metric = metric
        self.mode = mode
        self.optimizer = optimizer
        self.deepsupervision = deepsupervision
        self.clfhead = clfhead
        self.clf_threshold = clf_threshold
        self.dice_threshold = dice_threshold
        self.small_mask_threshold = small_mask_threshold
        self.early_stopping = early_stopping
        self.patience = 5
        
        self.scheduler = scheduler
        self.num_epochs = num_epochs
        self.parallel = parallel
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.save_last_model = save_last_model
        self.scheduler_step_per_epoch = scheduler_step_per_epoch
        self.learning_curves = dict()
        self.learning_curves['loss'],self.learning_curves['metric'] = dict(),dict()

        self.learning_curves['loss']['train'],self.learning_curves['loss']['val']=[],[]
        self.learning_curves['metric']['train'],self.learning_curves['metric']['val']=[],[]
        self.epoch_best=0
        self.best_val_loss = float('inf')
 
        self.best_val_metric = 0.0
        self.best_val_metric2 = 0.0
        self.best_model_wts = None
        self.checkpoint = None
        self.criterion2 = nn.BCEWithLogitsLoss().to(device)
        self.criterion_clf = nn.BCEWithLogitsLoss().to(device)
    def train_model(self):
        if self.device.type=='cpu':
            print('Start training the model on CPU')
        elif self.parallel and torch.cuda.device_count()>1:
            print(f'Start training the model on {torch.cuda.device_count()},{torch.cuda.get_device_name(torch.cuda.current_device())} in parallel')
            self.model = torch.nn.DataParallel(self.model)
        else:
            print(f'Start training the model on {torch.cuda.get_device_name(torch.cuda.current_device())}')
        self.model = self.model.to(self.device)
        with codecs.open('log.log','a') as up:
            up.write('\n\n')
            up.write(str(datetime.datetime.now()))
        log_cols = ['epoch', 'lr',
            'loss_trn', 'loss_val',
            'trn_score', 'val_score']
        log_df = pd.DataFrame(columns=log_cols, dtype=object)
        counter_ES = 0
        for epoch in range(self.num_epochs):
            print(f'Epoch {epoch}/{self.num_epochs -1}')
            print('_'*20)
            for phase in ['train','val']:
                if phase=='train':
                    self.model.train()
                else:
                    self.model.eval()
                phase_loss = 0.0
                phase_metric = 0.0
                with torch.set_grad_enabled(phase=='train'):
                    y_preds = []
                    y_trues = []
                    running_loss_trn = 0
                    trn_score_numer = 0
                    trn_score_denom = 0
                    batch = 0
                    for sample in self.dataloaders[phase]:
                        b,c,h,w = sample['image'].shape
                        if self.clfhead:
                            y_clf = sample['label'].to(self.device,dtype=torch.float)

                            if self.deepsupervision:
                                logits,logits_deeps,logits_clf = self.model(sample['image'].to(self.device))
                            else:
                                logits,logits_clf = self.model(sample['image'].to(self.device))
                        else:
                            if self.deepsupervision:
                                logits,logits_deeps = self.model(sample['image'].to(self.device))
                            else:
                                logits = self.model(sample['image'].to(self.device))
                        y_true = sample['mask'].to(self.device)

                        dice_numer, dice_denom = dice_sum_2((torch.sigmoid(logits)).detach().cpu().numpy(), 
                                                            y_true.detach().cpu().numpy(), 
                                                            dice_threshold=self.dice_threshold)
                        trn_score_numer += dice_numer 
                        trn_score_denom += dice_denom
                        loss = self.criterion2(logits,y_true)
                        loss += DiceLoss()(logits,y_true)
#                         print(y_clf,y_true,torch.sigmoid(logits),logits_clf,sample['image']) sample['image']<0
#                         loss += lovasz_hinge(logits.view(-1,h,w), y_true.view(-1,h,w))
                        if self.deepsupervision:
                            for logits_deep in logits_deeps:
                                loss += 0.1 * criterion_dice_non_empty(self.criterion2, logits_deep, y_true)
#                                 loss += 0.1 * criterion_lovasz_hinge_non_empty(self.criterion2, logits_deep, y_true)
                        if self.clfhead:
                            loss += self.criterion_clf(logits_clf.squeeze(-1),y_clf)
                        phase_loss += loss.item()
                        running_loss_trn += loss.item() * b
                        metric = dice_sum((torch.sigmoid(logits)).detach().cpu().numpy(), 
                                                            y_true.detach().cpu().numpy(), dice_threshold=self.dice_threshold,
                                                            small_mask_threshold=self.small_mask_threshold,)/b
                        phase_metric += metric.item()
                        with np.printoptions(precision=3, suppress=True):
                            print(f'batch: {batch} batch loss: {loss:.3f} \tmetric: {metric:.3f}')
#                             print(f'batch: {batch} batch loss: {loss:.3f} ')

#                         del input

                        # Backward pass + optimize only if in training phase:
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                            # zero the parameter gradients:
                            self.optimizer.zero_grad()

                            if self.scheduler and not self.scheduler_step_per_epoch:
                                self.scheduler.step()

                        del loss
                        batch += 1
#                         target = target.detach().cpu().numpy().tolist()
#                         output = output.detach().cpu().numpy().tolist()
#                         final_targets.extend(target)
#                         final_outputs.extend(output)
#                         if batch>10:
#                             break
                epoch_loss = running_loss_trn / len(self.dataloaders[phase])
                print('trn_score',trn_score_numer,trn_score_denom)
                phase_score = trn_score_numer.sum() / trn_score_denom.sum()
                phase_loss /= len(self.dataloaders[phase])
#                 phase_metric = self.metric(final_targets,final_outputs)
                self.learning_curves['loss'][phase].append(epoch_loss)
                self.learning_curves['metric'][phase].append(phase_score)

                print(f'{phase.upper()} loss: {phase_loss:.3f} \tavg_metric: {np.mean(phase_score):.3f}')

                # Save summary if it is the best val results so far:
                if phase == 'val':
                    if phase_score > self.best_val_metric:
                        self.best_val_metric = phase_score #update'
                        self.epoch_best = epoch
                        self.best_val_loss = epoch_loss
                        torch.save(self.model.state_dict(), str(self.output_dir)+f'fold{self.fold}/bestscore.pth') #save
                        print('model (best score) saved')
                    with codecs.open('log.log', 'a') as up:
                        up.write(f"Fold={self.fold}, Epoch={epoch}, Valid Dice={phase_score}\n")


            log_df.loc[epoch,log_cols] = np.array([epoch,
                                             [ group['lr'] for group in self.optimizer.param_groups ],
                                             self.learning_curves['loss']['train'][-1], self.learning_curves['loss']['val'][-1], 
                                             self.learning_curves['metric']['train'][-1], 
                                             self.learning_curves['metric']['val'][-1]], dtype='object')
            if self.early_stopping:
                if epoch_loss<self.best_val_loss:
                    self.best_val_metric = phase_score
                    self.best_val_loss  = epoch_loss
                    self.epoch_best  = epoch
                    counter_ES  = 0
                    torch.save(self.model.state_dict(), str(self.output_dir)+f'fold{self.fold}/bestloss.pth') 
                    print('model (best loss) saved')
                else:
                    counter_ES += 1
                if counter_ES > self.patience:
                    print('early stopping, epoch_best {:.0f}, loss_val_best {:.5f}, val_score_best{:.5f}'.format(self.epoch_best, self.best_val_loss, self.best_val_metric))
                    break
            # Adjust learning rate after val phase:
            if self.scheduler and self.scheduler_step_per_epoch:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(np.mean(phase_metric))
                else:
                    self.scheduler.step()
        log_df.to_csv(str(self.output_dir)+f'fold{self.fold}/log.csv', index=False)

        del self.model
        torch.cuda.empty_cache()
        gc.collect()
        
        print('')
    def save_results(self, path_to_dir):
        """"
        Save results in a directory. The method must be used after training.
        A short summary is stored in a csv file ('summary.csv'). Weights of the best model are stored in
        'best_model_weights.pt'. A checkpoint of the last epoch is stored in 'last_model_checkpoint.tar'. Two plots
        for the loss function and metric are stored in 'loss_plot.png' and 'metric_plot.png', respectively.
        Parameters
        ----------
        path_to_dir : str
            A path to the directory for storing all results.
        """

        path_to_dir = pathlib.Path(path_to_dir)

        # Check if the directory exists:

        # Write a short summary in a csv file:
        with open(path_to_dir / 'summary.csv', 'w', newline='', encoding='utf-8') as summary:
            summary.write(f'SUMMARY OF THE EXPERIMENT:\n\n')
            summary.write(f'BEST VAL EPOCH: {self.epoch_best}\n')
            summary.write(f'BEST VAL LOSS: {self.best_val_loss}\n')
            summary.write(f'BEST VAL metric: {self.best_val_metric}\n')

#         # Save best model weights:
#         torch.save(self.best_model_wts, path_to_dir / 'best_model_weights.pt')

#         # Save last model weights (checkpoint):
#         if self.save_last_model:
#             torch.save(self.checkpoint, path_to_dir / 'last_model_checkpoint.tar')

        # Save learning curves as pandas df:
        df_learning_curves = pd.DataFrame.from_dict({
            'loss_train': self.learning_curves['loss']['train'],
            'loss_val': self.learning_curves['loss']['val'],
            'metric_train': self.learning_curves['metric']['train'],
            'metric_val': self.learning_curves['metric']['val']
        })
        df_learning_curves.to_csv(path_to_dir / 'learning_curves.csv', sep=';')

        # Save learning curves' plots in png files:
        # Loss figure:
        plt.figure(figsize=(17.5, 10))
        plt.plot(range(self.num_epochs), self.learning_curves['loss']['train'], label='train')
        plt.plot(range(self.num_epochs), self.learning_curves['loss']['val'], label='val')
        plt.xlabel('Epoch', fontsize=20)
        plt.ylabel('Loss', fontsize=20)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.legend(fontsize=20)
        plt.grid()
        plt.savefig(path_to_dir / 'loss_plot.png', bbox_inches='tight')

        # metric figure:
        train_avg_metric = [np.mean(i) for i in self.learning_curves['metric']['train']]
        val_avg_metric = [np.mean(i) for i in self.learning_curves['metric']['val']]

        plt.figure(figsize=(17.5, 10))
        plt.plot(range(self.num_epochs), train_avg_metric, label='train')
        plt.plot(range(self.num_epochs), val_avg_metric, label='val')
        plt.xlabel('Epoch', fontsize=20)
        plt.ylabel('Avg metric', fontsize=20)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.legend(fontsize=20)
        plt.grid()
        plt.savefig(path_to_dir / 'metric_plot.png', bbox_inches='tight')

        print(f'All results have been saved in {path_to_dir}')

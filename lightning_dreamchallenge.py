
import os
import subprocess

import numpy as np
from collections import OrderedDict

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from torch import nn
from pytorch_lightning.utilities.cli import LightningCLI
from pytorch_lightning.utilities import cli as pl_cli
from pytorch_lightning.callbacks import ModelCheckpoint
from scipy.stats import pearsonr
import pytorch_lightning as pl

from dataloader import DreamChallengeDataModule

class Flatten(nn.Module):
    def forward(self, x):
        return torch.flatten(x, start_dim=1)

class NNHooks(pl.LightningModule):
    def __init__(self):
        super().__init__()
        
        self.example_input = torch.zeros(512, 4, 110).index_fill_(1, torch.tensor(2), 1)

    def training_step(self, batch, batch_idx):
        seq, y = batch
        y_hat = self(seq)
        y = torch.squeeze(y, 0)
        #print(torch.stack([y, y_hat], 1)[:5])

        # compute loss
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss)
        return {'loss': loss}
    
    def validation_step(self, batch, batch_idx):
        seq, y = batch
        y_hat = self(seq)
        y = torch.squeeze(y, 0)
        #print(torch.stack([y, y_hat], 1)[:5])

        # compute loss
        val_loss = F.mse_loss(y_hat, y)
        self.log('val_loss', val_loss)
        return {'val_loss': val_loss, 'pred': y_hat, 'true': y}

    def test_step(self, batch, batch_idx):
        seq, y = batch
        y_hat = self(seq)
        y = torch.squeeze(y, 0)

        # compute loss
        test_loss = F.mse_loss(y_hat, y)
        self.log('test_loss', test_loss)
        return {'test_loss': test_loss}

    def predict_step(self, batch, batch_idx):
        seq, y = batch
        y_hat = self(seq)

        return y_hat

    def validation_epoch_end(self, test_step_outputs):
        # plot a scatterplot to show correlation between prediction and target
        # collect outputs from each batch
        out_preds = []
        out_trues = []
        out_labels = []
        for outs in test_step_outputs:
            out_preds.append(outs['pred'])
            out_trues.append(outs['true'])
        
        # gather from ddp processes
        out_preds = self.all_gather(torch.stack(out_preds))
        out_trues = self.all_gather(torch.stack(out_trues))

        # plot figure in the main process
        if self.local_rank == 0: 
            out_preds = out_preds.detach().cpu().numpy().flatten()
            out_trues = out_trues.detach().cpu().numpy().flatten()

            print(out_preds[np.isnan(out_preds)])
            print(out_trues[np.isnan(out_trues)])

            fig = plt.figure(figsize=(12, 12))
            ax = sns.scatterplot(x=out_preds, y=out_trues)
            ax.set_xlim(left=0, right=8)
            ax.text(0.1, 0.8, f"pearsonr correlation efficient/p-value \n{pearsonr(out_preds, out_trues)}", transform=plt.gca().transAxes)
            self.logger.experiment.add_figure(f"Prediction vs True on validation dataset after epoch {self.current_epoch}", fig)

    def on_predict_epoch_end(self, predict_step_outputs):
        # collect outputs from each batch
        out_preds = []
        for outs in predict_step_outputs[0]: 
            out_preds.append(outs)
        
        # gather from ddp processes
        out_preds = self.all_gather(torch.cat(out_preds))

        # save
        if self.global_rank == 0:
            np.savetxt(os.path.join(self.logger.log_dir, "model_preds.txt"), out_preds.cpu().numpy().flatten(), fmt="%.6f")
        print(f"Saved model predictions into model_preds.txt")

@pl_cli.MODEL_REGISTRY
class ConvolutionalModel(NNHooks):
    def __init__(self, dropout=0.5):
        super().__init__()
        self.dropout = dropout

        self.model_strand_specific_forward = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv1d(4, 256, 31, padding=15)),
            ('relu1', nn.LeakyReLU()),
            ('batchnorm1', nn.BatchNorm1d(256)),
            ('conv2', nn.Conv1d(256, 256, 31, padding=15)),
            ('relu2', nn.LeakyReLU()),
            ('batchnorm2', nn.BatchNorm1d(256)),
        ]))
        self.model_strand_specific_reverse = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv1d(4, 256, 31, padding=15)),
            ('relu1', nn.LeakyReLU()),
            ('batchnorm1', nn.BatchNorm1d(256)),
            ('conv2', nn.Conv1d(256, 256, 31, padding=15)),
            ('relu2', nn.LeakyReLU()),
            ('batchnorm1', nn.BatchNorm1d(256)),
        ]))
        self.model_body = nn.Sequential(OrderedDict([
            ('conv3', nn.Conv1d(512, 256, 31, padding=15)),
            ('relu3', nn.LeakyReLU()),
            ('batchnorm3', nn.BatchNorm1d(256)),
            ('conv4', nn.Conv1d(256, 256, 31, padding=15)),
            ('relu4', nn.LeakyReLU()),
            ('batchnorm4', nn.BatchNorm1d(256)),
            ('faltten', Flatten()),
            ('dense1', nn.Linear(110*256, 256)),
            ('relu5', nn.LeakyReLU()),
            ('batchnorm5', nn.BatchNorm1d(256)),
            ('dropout1', nn.Dropout(self.dropout)),
            ('dense2', nn.Linear(256, 256)),
            ('relu6', nn.LeakyReLU()),
            ('batchnorm6', nn.BatchNorm1d(256)),
            ('dropout1', nn.Dropout(self.dropout))
        ]))
        self.model_head = nn.Sequential(OrderedDict([
            ('dense_head', nn.Linear(256, 1))
        ]))

    def forward(self, seq):
        seq = torch.squeeze(seq, 0)
        seq_revcomp = torch.flip(seq.detach().clone(), [1, 2])
        y_hat_for = self.model_strand_specific_forward(seq)
        y_hat_rev = self.model_strand_specific_reverse(seq_revcomp)
        y_hat = torch.cat([y_hat_for, y_hat_rev], dim=1)
        y_hat = self.model_body(y_hat)
        y_hat = self.model_head(y_hat)

        return y_hat

@pl_cli.MODEL_REGISTRY
class ConvolutionalModelAdj(NNHooks):
    def __init__(self, dropout=0.5, num_conv=2, dilation=1):
        super().__init__()
        self.num_conv = num_conv
        self.dilation = dilation
        self.dropout = dropout

        self.model_strand_specific_forward = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv1d(4, 256, 31, padding=15)),
            ('relu1', nn.LeakyReLU()),
            ('batchnorm1', nn.BatchNorm1d(256)),
            ('conv2', nn.Conv1d(256, 256, 31, padding=15)),
            ('relu2', nn.LeakyReLU()),
            ('batchnorm2', nn.BatchNorm1d(256)),
        ]))
        self.model_strand_specific_reverse = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv1d(4, 256, 31, padding=15)),
            ('relu1', nn.LeakyReLU()),
            ('batchnorm1', nn.BatchNorm1d(256)),
            ('conv2', nn.Conv1d(256, 256, 31, padding=15)),
            ('relu2', nn.LeakyReLU()),
            ('batchnorm1', nn.BatchNorm1d(256)),
        ]))
        conv_repeat_dict = OrderedDict([])
        for i in range(self.num_conv):
            conv_repeat_dict[f"conv_repeat_{i}"] = nn.Sequential(OrderedDict([
                                                    (f'conv_repeat_{i}', nn.Conv1d(512, 512, 31, padding=15*self.dilation, dilation=self.dilation)),
                                                    (f'relu_repeat_{i}', nn.LeakyReLU()),
                                                    (f'batchnorm_repeat_{i}', nn.BatchNorm1d(512)),
                                                    ]))
        self.model_conv_repeat = nn.Sequential(conv_repeat_dict)
        self.model_body = nn.Sequential(OrderedDict([
            ('faltten', Flatten()),
            ('dense1', nn.Linear(110*512, 256)),
            ('relu5', nn.LeakyReLU()),
            ('batchnorm5', nn.BatchNorm1d(256)),
            ('dropout1', nn.Dropout(self.dropout)),
            ('dense2', nn.Linear(256, 256)),
            ('relu6', nn.LeakyReLU()),
            ('batchnorm6', nn.BatchNorm1d(256)),
            ('dropout1', nn.Dropout(self.dropout))
        ]))
        self.model_head = nn.Sequential(OrderedDict([
            ('dense_head', nn.Linear(256, 1))
        ]))

    def forward(self, seq):
        seq = torch.squeeze(seq, 0)
        seq_revcomp = torch.flip(seq.detach().clone(), [1, 2])
        y_hat_for = self.model_strand_specific_forward(seq)
        y_hat_rev = self.model_strand_specific_reverse(seq_revcomp)
        y_hat = torch.cat([y_hat_for, y_hat_rev], dim=1)
        y_hat = self.model_conv_repeat(y_hat)
        y_hat = self.model_body(y_hat)
        y_hat = self.model_head(y_hat)

        return y_hat

@pl_cli.MODEL_REGISTRY
class ConvolutionalModelHybrid(NNHooks):
    def __init__(self, dropout=0.5, num_hybrid_conv=2):
        super().__init__()
        self.num_hybrid_conv=num_hybrid_conv
        self.dropout = dropout

        self.model_strand_specific_forward = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv1d(4, 256, 31, padding=15)),
            ('relu1', nn.LeakyReLU()),
            ('batchnorm1', nn.BatchNorm1d(256)),
            ('conv2', nn.Conv1d(256, 256, 31, padding=15)),
            ('relu2', nn.LeakyReLU()),
            ('batchnorm2', nn.BatchNorm1d(256)),
        ]))
        self.model_strand_specific_reverse = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv1d(4, 256, 31, padding=15)),
            ('relu1', nn.LeakyReLU()),
            ('batchnorm1', nn.BatchNorm1d(256)),
            ('conv2', nn.Conv1d(256, 256, 31, padding=15)),
            ('relu2', nn.LeakyReLU()),
            ('batchnorm1', nn.BatchNorm1d(256)),
        ]))
        hybrid_conv_repeat_dict = OrderedDict([])
        for i in range(self.num_hybrid_conv):
            hybrid_conv_repeat_dict[f"hybrid_conv_repeat_{i}"] = nn.Sequential(OrderedDict([
                (f'hybrid_conv_{i}_dilation1', nn.Conv1d(512, 512, 31, padding=15*1, dilation=1)),
                (f'relu_{i}_1', nn.LeakyReLU()),
                (f'batchnorm_{i}_1', nn.BatchNorm1d(512)),
                (f'hybrid_conv_{i}_dilation3', nn.Conv1d(512, 512, 31, padding=15*3, dilation=3)),
                (f'relu_{i}_3', nn.LeakyReLU()),
                (f'batchnorm_{i}_3', nn.BatchNorm1d(512)),
                (f'hybrid_conv_{i}_dilation5', nn.Conv1d(512, 512, 31, padding=15*5, dilation=5)),
                (f'relu_{i}_5', nn.LeakyReLU()),
                (f'batchnorm_{i}_5', nn.BatchNorm1d(512)),
            ]))
        self.model_hybrid_conv_repeat = nn.Sequential(hybrid_conv_repeat_dict)
        self.model_body = nn.Sequential(OrderedDict([
            ('faltten', Flatten()),
            ('dense1', nn.Linear(110*512, 256)),
            ('relu5', nn.LeakyReLU()),
            ('batchnorm5', nn.BatchNorm1d(256)),
            ('dropout1', nn.Dropout(self.dropout)),
            ('dense2', nn.Linear(256, 256)),
            ('relu6', nn.LeakyReLU()),
            ('batchnorm6', nn.BatchNorm1d(256)),
            ('dropout1', nn.Dropout(self.dropout))
        ]))
        self.model_head = nn.Sequential(OrderedDict([
            ('dense_head', nn.Linear(256, 1))
        ]))

    def forward(self, seq):
        seq = torch.squeeze(seq, 0)
        seq_revcomp = torch.flip(seq.detach().clone(), [1, 2])
        y_hat_for = self.model_strand_specific_forward(seq)
        y_hat_rev = self.model_strand_specific_reverse(seq_revcomp)
        y_hat = torch.cat([y_hat_for, y_hat_rev], dim=1)
        y_hat = self.model_hybrid_conv_repeat(y_hat)
        y_hat = self.model_body(y_hat)
        y_hat = self.model_head(y_hat)

        return y_hat

class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_optimizer_args(torch.optim.Adam)

    def after_fit(self):
        for cb in self.trainer.callbacks:
            if isinstance(cb, ModelCheckpoint):
                newname = os.path.join(os.path.dirname(cb.best_model_path), os.path.splitext(os.path.basename(cb.best_model_path))[0] + "_best.ckpt")
                subprocess.check_call(["cp", cb.best_model_path, newname])
            break

def main():
    import os
    print(os.environ['MASTER_ADDR'])
    print(os.environ['NODE_RANK'])
    cli = MyLightningCLI(datamodule_class=DreamChallengeDataModule, auto_registry=True, save_config_overwrite=True)

if __name__ == '__main__':
    main()

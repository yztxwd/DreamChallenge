
import args_parse

FLAGS = args_parse.parse_common_options(
    log_steps=200,
    datadir='./shards/',
    logdir='./conv_model/',
    batch_size=512,
    momentum=0.5,
    lr=0.001,
    num_epochs=2,
    num_workers=32)

import os
import shutil
import sys
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch_xla
import torch_xla.debug.metrics as met
import torch_xla.distributed.parallel_loader as pl
import torch_xla.utils.utils as xu
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.test.test_utils as test_utils
import webdataset as wds
import matplotlib.pyplot as plt
import seaborn as sns
from dataloader import DreamChallengeDataModule
from tqdm import tqdm
from torchmetrics import MeanSquaredError

class ConvolutionalModel(nn.Module):
    def __init__(self):
        super(ConvolutionalModel, self).__init__()

        self.model_strand_specific_forward = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv1d(4, 256, 31, padding=15)),
            ('relu1', nn.ReLU()),
            ('conv2', nn.Conv1d(256, 256, 31, padding=15)),
            ('relu2', nn.ReLU())
        ]))
        self.model_strand_specific_reverse = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv1d(4, 256, 31, padding=15)),
            ('relu1', nn.ReLU()),
            ('conv2', nn.Conv1d(256, 256, 31, padding=15)),
            ('relu2', nn.ReLU())
        ]))
        self.model_body = nn.Sequential(OrderedDict([
            ('conv3', nn.Conv1d(512, 256, 31, padding=15)),
            ('relu3', nn.ReLU()),
            ('conv4', nn.Conv1d(256, 256, 31, padding=15)),
            ('relu4', nn.ReLU()),
            ('faltten', nn.Flatten()),
            ('dense1', nn.Linear(110*256, 256)),
            ('relu5', nn.ReLU()),
            ('dropout1', nn.Dropout(0.2)),
            ('dense2', nn.Linear(256, 256)),
            ('relu6', nn.ReLU()),
            ('dropout1', nn.Dropout(0.2))
        ]))
        self.model_head = nn.Sequential(OrderedDict([
            ('dense_head', nn.Linear(256, 1))
        ]))

        self.example_input = torch.zeros(512, 4, 500).index_fill_(1, torch.tensor(2), 1)

    def forward(self, seq):
        seq_revcomp = torch.flip(seq.detach().clone(), [1, 2])
        y_hat_for = self.model_strand_specific_forward(seq)
        y_hat_rev = self.model_strand_specific_reverse(seq_revcomp)
        y_hat = torch.cat([y_hat_for, y_hat_rev], dim=1)
        y_hat = self.model_body(y_hat)
        y_hat = self.model_head(y_hat)

        return y_hat

def plot_scatter(x, y, writer, lim=20):
  fig = plt.figure(12, 12)
  ax = sns.scatterplot(x=x, y=y)
  ax.set_xlim(left=0, right=lim)
  ax.set_ylim(bottom=0, right=lim)
  writer.add_figure("Prediction on test data", fig)  

def _train_update(device, x, loss, tracker, writer):
  loss_item = loss.item()
  test_utils.print_training_update(
      device,
      x,
      loss_item,
      tracker.rate(),
      tracker.global_rate(),
      summary_writer=writer)
  test_utils.write_to_summary(
      writer,
      dict_to_write={
        "loss": loss_item
      })

def train_model(flags, **kwargs):
  torch.manual_seed(1)

  datamodule = DreamChallengeDataModule(data_dir=flags.datadir, batch_size=flags.batch_size, num_workers=flags.num_workers)
  datamodule.setup()
  train_loader = datamodule.train_dataloader()
  val_loader = datamodule.val_dataloader()

  # Scale learning rate to num cores
  lr = flags.lr * xm.xrt_world_size()

  device = xm.xla_device()
  model = ConvolutionalModel().to(device)
  writer = None
  if xm.is_master_ordinal():
    writer = test_utils.get_summary_writer(flags.logdir)
    writer.add_graph(model, model.example_input.to(device))
  #optimizer = optim.SGD(model.parameters(), lr=lr, momentum=flags.momentum)
  optimizer = optim.Adam(model.parameters(), lr=lr)
  loss_fn = nn.MSELoss()

  def train_loop_fn(epoch, loader):
    tracker = xm.RateTracker()
    model.train()
    pbar = tqdm(enumerate(loader))
    for step, (data, target) in pbar:
      optimizer.zero_grad()
      data = torch.squeeze(data, 0)
      target = torch.squeeze(target, 0)
      output = model(data)
      loss = loss_fn(output, target)
      loss.backward()
      xm.optimizer_step(optimizer)
      tracker.add(flags.batch_size)
      pbar.set_description("Loss %s" % loss)
      if step % flags.log_steps == 0:
        xm.add_step_closure(
          _train_update,
          args=(device, step, loss, tracker, writer),
          run_async=flags.async_closures
        )
    if xm.is_master_ordinal():
      torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss.detach().cpu().numpy()
      }, os.path.join(flags.logdir, f"checkpoint-epoch{epoch}.ckpt"))
      
  def val_loop_fn(loader):
    model.eval()
    metrics = MeanSquaredError()
    pbar = tqdm(enumerate(loader))
    for step, (data, target) in pbar:
      data = torch.squeeze(data, 0)
      target = torch.squeeze(target, 0)
      pred = model(data)
      metrics.update(pred, target)

    mse = metrics.compute()  
    mse = xm.mesh_reduce('val_mse', mse, np.mean)
    return mse

  def test_loop_fn(loader):
    model.eval()
    metrics = MeanSquaredError()
    pbar = tqdm(enumerate(loader))
    targets = []
    preds = []
    for step, (data, target) in pbar:
      data = torch.squeeze(data, 0)
      target = torch.squeeze(target, 0)
      pred = model(data)
      metrics.update(pred, target)
      targets.append(target)
      preds.append(pred)
    # collect data from all processes
    targets = xm.all_gather(torch.stack(targets))
    preds = xm.all_gather(torch.stack(preds))
    if xm.is_master_ordinal(): plot_scatter(targets, preds, writer)
    # compute the MSE per-process then reduce
    mse = metrics.compute()  
    mse = xm.mesh_reduce('test_mse', mse, np.mean)
    return mse

  train_device_loader = pl.MpDeviceLoader(train_loader, device)
  val_device_loader = pl.MpDeviceLoader(val_loader, device)
  mse, min_mse = 1e3, 1e3
  best_epoch = -1
  for epoch in range(1, flags.num_epochs + 1):
    xm.master_print('Epoch {} train begin {}'.format(epoch, test_utils.now()))
    train_loop_fn(epoch, train_device_loader)
    xm.master_print('Epoch {} train end {}'.format(epoch, test_utils.now()))

    mse = val_loop_fn(val_device_loader)
    xm.master_print('Epoch {} test end {}, MSE={:.2f}'.format(
        epoch, test_utils.now(), mse))
    if mse < min_mse:
      min_mse = mse
      best_epoch = epoch
    test_utils.write_to_summary(
        writer,
        epoch,
        dict_to_write={'MSE/val': mse},
        write_xla_metrics=True)
    if flags.metrics_debug:
      xm.master_print(met.metrics_report())

  test_utils.close_summary_writer(writer)
  xm.master_print(f'Min MSE: {min_mse:.2f} from Epoch {best_epoch}')
  return min_mse, best_epoch

def _mp_fn(index, flags):
  torch.set_default_tensor_type('torch.FloatTensor')
  mse, best_epoch = train_model(flags)
  if flags.tidy and os.path.isdir(flags.datadir):
    shutil.rmtree(flags.datadir)

if __name__ == '__main__':
  xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=FLAGS.num_cores)

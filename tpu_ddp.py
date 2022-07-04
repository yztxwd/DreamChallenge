
import args_parse

FLAGS = args_parse.parse_common_options(
    log_steps=200,
    datadir='./shards/',
    logdir='./conv_model/',
    batch_size=512,
    momentum=0.5,
    lr=0.0001,
    num_epochs=100,
    num_workers=32,
    opts=[('--patience', {'type': int, 'default': 5}),
          ('--train_epochs', {'type': int, 'default': None})])

import os
import shutil
import sys
import subprocess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_xla
import torch_xla.debug.metrics as met
import torch_xla.distributed.parallel_loader as pl
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.test.test_utils as test_utils
import webdataset as wds
import matplotlib.pyplot as plt
import seaborn as sns
from dataloader import DreamChallengeDataModule
from tqdm import tqdm
from torchmetrics import MeanSquaredError

import models

def _train_update(device, x, loss, tracker, writer):
  loss_item = loss.item()
  #test_utils.print_training_update(
  #    device,
  #    x
  #    loss_item,
  #    tracker.rate(),
  #    tracker.global_rate(),
  #    summary_writer=writer)
  test_utils.write_to_summary(
      writer,
      dict_to_write={
        "loss": loss_item
      })

def train_model(flags, **kwargs):
  torch.manual_seed(1)

  print(f"Device {xm.get_ordinal()} in world size of {xm.xrt_world_size()}")
  num_data_instances = xm.xrt_world_size() * flags.num_workers
  train_epochs = flags.train_epochs if flags.train_epochs is not None else 5391406//num_data_instances//flags.batch_size
  datamodule = DreamChallengeDataModule(data_dir=flags.datadir, train_epochs=train_epochs, batch_size=flags.batch_size, num_workers=flags.num_workers)
  datamodule.setup()
  train_loader = datamodule.train_dataloader()
  val_loader = datamodule.val_dataloader()

  # Scale learning rate to num cores
  lr = flags.lr * xm.xrt_world_size()

  model = models.ConvolutionalModel()
  writer = None
  if xm.is_master_ordinal():
    writer = test_utils.get_summary_writer(flags.logdir)
    writer.add_graph(model, model.example_input)
  device = xm.xla_device()
  model = model.to(device)
  #optimizer = optim.SGD(model.parameters(), lr=lr, momentum=flags.momentum)
  optimizer = optim.Adam(model.parameters(), lr=lr)
  loss_fn = nn.MSELoss()

  def train_loop_fn(epoch, loader):
    tracker = xm.RateTracker()
    model.train()
    pbar = tqdm(enumerate(loader))
    for step, (data, target) in pbar:
      data = torch.squeeze(data, 0)
      target = torch.squeeze(target, 0)
      optimizer.zero_grad()
      output = model(data)
      loss = loss_fn(output, target)
      loss.backward()
      #xm.optimizer_step(optimizer)
      xm.reduce_gradients(optimizer)
      torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5, error_if_nonfinite=True)
      optimizer.step()
      tracker.add(flags.batch_size)
      if xm.is_master_ordinal():
        pbar.set_description("Loss %s" % loss.item())
        #if step % flags.log_steps == 0:
        #  xm.add_step_closure(
        #    _train_update,
        #    args=(device, step, loss, tracker, writer),
        #    run_async=flags.async_closures
        #  )
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
      if xm.is_master_ordinal(): 
        if step == 0:
          print(torch.stack([target, pred], 1)[:10])

    mse = metrics.compute()  
    mse = xm.mesh_reduce('val_mse', mse, np.mean)
    return mse

  train_device_loader = pl.MpDeviceLoader(train_loader, device)
  val_device_loader = pl.MpDeviceLoader(val_loader, device)
  # Early stopping
  mse, min_mse = 1e3, 1e3
  best_epoch = -1
  patience = FLAGS.patience
  trigger_times = 0
  for epoch in range(1, flags.num_epochs + 1):
    xm.master_print('Epoch {} train begin {}'.format(epoch, test_utils.now()))
    train_loop_fn(epoch, train_device_loader)
    xm.master_print('Epoch {} train end {}'.format(epoch, test_utils.now()))

    mse = val_loop_fn(val_device_loader)
    xm.master_print('Epoch {} end {}, Val MSE={:.2f}'.format(
        epoch, test_utils.now(), mse))
    test_utils.write_to_summary(
        writer,
        epoch,
        dict_to_write={'MSE/val': mse},
        write_xla_metrics=True)
    if flags.metrics_debug:
      xm.master_print(met.metrics_report())

    # Early stopping
    if mse > min_mse:
        trigger_times += 1
        #print('Trigger Times:', trigger_times)
        if trigger_times >= patience:
            print('Early stopping!\nStart to test process.')
            break
    else:
        #print('trigger times: 0')
        trigger_times = 0

        min_mse = mse
        best_epoch = epoch

  test_utils.close_summary_writer(writer)
  xm.master_print(f'Min MSE: {min_mse:.2f} from Epoch {best_epoch}')
  if xm.is_master_ordinal(): subprocess.run(['cp', os.path.join(flags.logdir, f"checkpoint-epoch{best_epoch}.ckpt"), os.path.join(flags.logdir, f"checkpoint-epoch{best_epoch}-best.ckpt")], shell=True)

  return min_mse, best_epoch
  
def _mp_fn(index, flags):
  torch.set_default_tensor_type('torch.FloatTensor')
  mse, best_epoch = train_model(flags)
  if flags.tidy and os.path.isdir(flags.datadir):
    shutil.rmtree(flags.datadir)

if __name__ == '__main__':
  xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=FLAGS.num_cores)
